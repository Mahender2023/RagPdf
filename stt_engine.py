# stt_engine.py
import streamlit as st 
import pyaudio
import queue
import logging
import threading 
from google.cloud import speech
import google.api_core.exceptions
import grpc 

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms of audio

# --- Logging Configuration ---
# Basic logging config. For more advanced, consider using Streamlit's logger or a dedicated setup.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
        # You could add logging.FileHandler("stt_engine.log") here as well
    ]
)
logger = logging.getLogger(__name__)


# --- MicrophoneStream Class ---
class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._audio_interface = None
        self._stream = None
        self._audio_queue = queue.Queue()
        self._closed = True # Initialize as closed
        self.error_message = None
        logger.info("MicrophoneStream instance created.")

    def open_stream(self):
        if not self._closed:
            logger.warning("Attempted to open an already open stream.")
            return True # Or False, depending on desired behavior

        try:
            self._audio_interface = pyaudio.PyAudio()
            self._stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._rate,
                input=True,
                frames_per_buffer=self._chunk,
                stream_callback=self._fill_buffer,
            )
            self._closed = False
            self.error_message = None
            logger.info("Microphone stream opened successfully.")
            return True
        except Exception as e:
            self.error_message = f"Failed to open microphone: {e}"
            logger.error(self.error_message, exc_info=True)
            self.close_stream() # Ensure cleanup if open fails
            return False

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        if self._closed: # Stop collecting if closed
            return None, pyaudio.paComplete
        self._audio_queue.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        logger.info("Audio generator started.")
        while not self._closed:
            try:
                # Use a timeout to allow the loop to check self._closed periodically
                chunk = self._audio_queue.get(block=True, timeout=0.1)
                if chunk is None: # Sentinel value from close_stream
                    logger.info("Audio generator received None (sentinel), stopping.")
                    return
                yield chunk
            except queue.Empty:
                # Timeout occurred, loop back to check self._closed
                if self._closed:
                    logger.info("Audio generator detected stream closed during timeout, stopping.")
                    return
                continue
            except Exception as e:
                logger.error(f"Audio generator error: {e}", exc_info=True)
                return # Stop generator on unexpected error
        logger.info("Audio generator finished (stream was closed).")


    def close_stream(self):
        if self._closed:
            logger.info("close_stream() called, but stream was already closed or not opened.")
            return

        logger.info("close_stream() called.")
        self._closed = True # Signal to stop _fill_buffer and generator
        self._audio_queue.put(None) # Send sentinel to unblock generator if waiting

        try:
            if self._stream is not None:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
                logger.info("PyAudio stream stopped and closed.")
        except Exception as e:
            logger.error(f"Error closing PyAudio stream: {e}", exc_info=True)
        finally:
            self._stream = None

        try:
            if self._audio_interface is not None:
                self._audio_interface.terminate()
                logger.info("PyAudio interface terminated.")
        except Exception as e:
            logger.error(f"Error terminating PyAudio interface: {e}", exc_info=True)
        finally:
            self._audio_interface = None

        # Clear any remaining items from the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("MicrophoneStream resources released and queue cleared.")

# --- STT Processing Function ---
def process_audio_stream(client: speech.SpeechClient,
                         streaming_config: speech.StreamingRecognitionConfig,
                         audio_generator,
                         stop_event: threading.Event,
                         ui_queue: queue.Queue):
    logger.info("STT processing thread started.")

    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in audio_generator
    )

    try:
        logger.info("Calling client.streaming_recognize...")
        responses = client.streaming_recognize(streaming_config, requests, timeout=300) # 5 min timeout

        for response_count, response in enumerate(responses):
            if stop_event.is_set():
                logger.info(f"Stop event detected in STT response loop (after {response_count} responses). Breaking.")
                break

            if response.speech_event_type == speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_UNSPECIFIED:
                 pass # Normal, no specific event

            if response.error and response.error.code != 0 : # Ensure code is not 0 (OK)
                logger.error(f"STT API Error: {response.error.message} (Code: {response.error.code})")
                ui_queue.put({"type": "error", "message": f"STT API Error ({response.error.code}): {response.error.message}"})
                break # Stop processing on API error

            if not response.results:
                # logger.debug("STT: No results in this response.") # Can be verbose
                continue

            result = response.results[0]
            if not result.alternatives:
                # logger.debug("STT: No alternatives in this result.") # Can be verbose
                continue

            transcript = result.alternatives[0].transcript

            if result.is_final:
                logger.info(f"STT Final transcript part: '{transcript}'")
                ui_queue.put({"type": "final", "text": transcript})
            else:
                # logger.debug(f"STT Interim transcript: '{transcript}'") # Can be very verbose
                ui_queue.put({"type": "interim", "text": transcript})

        logger.info("STT response loop finished or broken.")

    except StopIteration:
        logger.info("STT: Audio generator stopped (StopIteration), common if mic stream closed.")
    except google.api_core.exceptions.OutOfRange as e:
        logger.warning(f"STT: Stream ended by server (OutOfRange), often due to silence or max duration: {e}")
        ui_queue.put({"type": "status", "message": f"Stream limit reached or long silence."})
    except google.api_core.exceptions.Cancelled as e:
        logger.info(f"STT: Streaming cancelled by client or server: {e}")
        ui_queue.put({"type": "status", "message": "Streaming cancelled."})
    except grpc.RpcError as e:
        code = e.code()
        details = e.details()
        logger.error(f"STT: gRPC error during streaming: {code} - {details}")
        ui_queue.put({"type": "error", "message": f"STT Network/gRPC Error: {details} (Code: {code})"})
    except Exception as e:
        logger.error(f"STT: Unexpected error in process_audio_stream: {e}", exc_info=True)
        ui_queue.put({"type": "error", "message": f"STT Processing Error: {str(e)}"})
    finally:
        logger.info("STT processing thread performing cleanup and signaling done.")
        # stop_event might already be set by the main thread.
        # If the STT thread exits due to an internal error, it's good practice to ensure it's set.
        stop_event.set()
        ui_queue.put({"type": "thread_done"})
        logger.info("STT processing thread finished.")