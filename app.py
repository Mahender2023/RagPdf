# app.py
import streamlit as st
import os
import queue
import threading
import time
import json
from datetime import datetime
from dotenv import load_dotenv
import logging 

# Import engines/modules
import stt_engine
import rag_engine # Assuming this handles its own logging
from google.cloud import speech # For STT client
import google.generativeai as genai

# --- Initial Configuration ---
load_dotenv()

# Configure logging for app.py
# Using Streamlit's logger is often a good practice if you want to integrate with its mechanisms
# For simplicity here, using standard logging.
app_logger = logging.getLogger(__name__) # Now 'logging' is defined
app_logger.setLevel(logging.INFO)
# Ensure handlers are set up if not already by other modules
if not app_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app_logger.addHandler(handler)


# Constants
STT_RATE = stt_engine.RATE
STT_CHUNK = stt_engine.CHUNK
CHAT_HISTORY_FILE = "chat_history.json"
DEBUG_MODE = os.environ.get("STREAMLIT_DEBUG_MODE", "False").lower() == "true"

# --- Session State Initialization ---
def initialize_session_state():
    default_states = {
        # RAG States
        "rag_chat_history": [],
        "document_processed": False,
        "current_language_display": "English",
        "rag_status_message": "",
        "rag_api_key_configured": bool(rag_engine.gemini_chat_model), # Assuming rag_engine has this
        "debug_retrieved_chunks": [],
        "all_processed_chunks_for_debug": [],

        "active_query_text": "",

        # STT States
        "stt_is_recording": False,
        "stt_full_transcript": "",
        "stt_interim_transcript": "",
        "stt_google_client": None,
        "stt_audio_thread": None,
        "stt_microphone_stream_obj": None,
        "stt_stop_event_thread": threading.Event(),
        "stt_ui_update_queue": queue.Queue(),
        "stt_status_message": "Click 'Start Recording' when ready.",
        "stt_error": None,
        "stt_credentials_ok": False,
        "submit_rag_query_now": False,
        "stt_transcript_to_populate": None,
        "clear_query_box_on_next_run": False,
    }
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.rag_chat_history and os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.rag_chat_history = json.load(f)
        except Exception as e:
            app_logger.error(f"Error loading RAG chat history: {e}", exc_info=True)
            st.session_state.rag_chat_history = []

# --- Helper Functions ---
def save_rag_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.rag_chat_history, f, indent=4, ensure_ascii=False)
    except Exception as e:
        app_logger.error(f"Error saving RAG chat history: {e}", exc_info=True)

def initialize_stt_client():
    if st.session_state.stt_google_client is None:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not cred_path or not os.path.exists(cred_path):
            st.session_state.stt_status_message = "STT Error: GOOGLE_APPLICATION_CREDENTIALS not found or invalid."
            st.session_state.stt_credentials_ok = False
            app_logger.error(st.session_state.stt_status_message)
            return False
        try:
            st.session_state.stt_google_client = speech.SpeechClient()
            st.session_state.stt_status_message = "STT Client ready. Click 'Start Recording'."
            st.session_state.stt_credentials_ok = True
            app_logger.info("Google STT SpeechClient initialized successfully.")
            return True
        except Exception as e:
            st.session_state.stt_status_message = f"STT Error: Google STT client init failed: {e}"
            st.session_state.stt_credentials_ok = False
            app_logger.error(st.session_state.stt_status_message, exc_info=True)
            return False
    return st.session_state.stt_credentials_ok

# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Voice RAG Assistant", layout="wide")
    st.title("🎙️ RAGpdf Q&A")

    initialize_session_state()

    stt_ready = initialize_stt_client()
    rag_ready = st.session_state.rag_api_key_configured

    if not stt_ready:
        st.error(f"🔴 STT Client Not Ready: {st.session_state.stt_status_message}. Check GOOGLE_APPLICATION_CREDENTIALS.")
    if not rag_ready:
        st.error("🔴 RAG Engine Not Ready: GOOGLE_API_KEY for Gemini not configured. Check .env file.")

    if not stt_ready or not rag_ready:
        st.warning("Please configure API keys in your .env file and restart.")
        st.stop()

    # --- Section 1: Document Processing (Assuming no changes needed here from your previous version) ---
    st.header("📄 Document Setup")
    st.subheader("1. Upload and Process Document")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, JPG, or PNG:", type=["pdf", "docx", "jpg", "jpeg", "png"],
        key="file_uploader_integrated"
    )

    if st.button("⚙️ Process Document", key="process_doc_btn_integrated", disabled=not uploaded_file):
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                rag_engine.global_vector_store.reset()
                st.session_state.document_processed = False
                st.session_state.debug_retrieved_chunks = []
                st.session_state.all_processed_chunks_for_debug = []
                original_filename = uploaded_file.name
                file_name_lower = original_filename.lower()
                source_texts_for_chunking = []
                processed_doc_type = ""

                if file_name_lower.endswith(".pdf"):
                    source_texts_for_chunking = rag_engine.load_pdf_pages_pymupdf(uploaded_file)
                    processed_doc_type = "PDF"
                elif file_name_lower.endswith(".docx"):
                    docx_full_text = rag_engine.load_docx_text(uploaded_file)
                    if docx_full_text: source_texts_for_chunking = [docx_full_text]
                    processed_doc_type = "DOCX"
                elif file_name_lower.endswith((".jpg", ".jpeg", ".png")):
                    image_description = rag_engine.get_image_description_gemini(uploaded_file, original_filename)
                    if image_description:
                        source_texts_for_chunking = [f"Image: '{original_filename}'. Description:\n{image_description}"]
                    processed_doc_type = "Image"

                if not source_texts_for_chunking or not any(item.strip() for item in source_texts_for_chunking):
                    st.session_state.rag_status_message = f"Warning: No text/description extracted from '{original_filename}'."
                else:
                    all_final_chunks = []
                    for text_block in source_texts_for_chunking:
                        chunks_from_block = rag_engine.get_text_chunks(text_block)
                        if chunks_from_block: all_final_chunks.extend(chunks_from_block)
                    st.session_state.all_processed_chunks_for_debug = all_final_chunks

                    if not all_final_chunks:
                        st.session_state.rag_status_message = "Warning: Document processed, but no meaningful chunks created."
                    else:
                        processed_chunks_count = 0
                        for text_chunk_to_embed in all_final_chunks:
                            emb = rag_engine.get_google_embedding(text_chunk_to_embed, task_type="RETRIEVAL_DOCUMENT")
                            if emb:
                                rag_engine.global_vector_store.add(text_chunk_to_embed, emb)
                                processed_chunks_count += 1
                        if rag_engine.global_vector_store.index.ntotal > 0:
                            st.session_state.document_processed = True
                            st.session_state.rag_status_message = f"✅ Success: {processed_doc_type} '{original_filename}' processed. {processed_chunks_count} of {len(all_final_chunks)} chunks vectorized."
                        else:
                            st.session_state.rag_status_message = f"⚠️ Warning: {processed_doc_type} '{original_filename}' processed, but no content was vectorized."
            st.rerun()

    if st.session_state.rag_status_message:
        if "Success" in st.session_state.rag_status_message: st.success(st.session_state.rag_status_message)
        elif "Warning" in st.session_state.rag_status_message: st.warning(st.session_state.rag_status_message)
        else: st.info(st.session_state.rag_status_message)
        st.session_state.rag_status_message = ""

    st.divider()

    # --- Section 2: Ask Your Question ---
    st.header("💬 Ask Your Question")

    if not st.session_state.document_processed:
        st.warning("Please upload and process a document above before asking questions.")
    else:
        if st.session_state.get("clear_query_box_on_next_run"):
            st.session_state.active_query_text = ""
            st.session_state.clear_query_box_on_next_run = False

        if st.session_state.get("stt_transcript_to_populate") is not None:
            st.session_state.active_query_text = st.session_state.stt_transcript_to_populate
            st.session_state.stt_transcript_to_populate = None

        lang_display_names = list(rag_engine.LANGUAGE_OPTIONS_MAP.keys())
        try:
            current_lang_index = lang_display_names.index(st.session_state.current_language_display)
        except ValueError:
            current_lang_index = 0

        selected_language_display = st.selectbox(
            "Response Language:", options=lang_display_names, index=current_lang_index,
            key="rag_lang_select"
        )
        if selected_language_display != st.session_state.current_language_display:
            st.session_state.current_language_display = selected_language_display
            st.rerun()

        st.markdown("#### 🎤 Speak your query")
        st.caption("(Ensure microphone access is allowed in your browser.)")
        can_record = st.session_state.document_processed and stt_ready

        col_start_rec, col_stop_rec = st.columns(2)
        with col_start_rec:
            if st.button("▶️ Start Recording", disabled=st.session_state.stt_is_recording or not can_record, key="start_rec_btn", use_container_width=True):
                app_logger.info("Start Recording button clicked.")
                st.session_state.stt_is_recording = True
                st.session_state.stt_full_transcript = ""
                st.session_state.stt_interim_transcript = ""
                st.session_state.stt_status_message = "Initializing microphone..."
                st.session_state.stt_error = None
                st.session_state.stt_stop_event_thread.clear()
                while not st.session_state.stt_ui_update_queue.empty():
                    try: st.session_state.stt_ui_update_queue.get_nowait()
                    except queue.Empty: break

                mic_stream = stt_engine.MicrophoneStream(STT_RATE, STT_CHUNK)
                if not mic_stream.open_stream():
                    st.session_state.stt_is_recording = False
                    st.session_state.stt_status_message = f"Mic Error: {mic_stream.error_message or 'Could not open microphone.'}"
                    st.session_state.stt_error = st.session_state.stt_status_message
                    app_logger.error(f"Failed to open microphone: {st.session_state.stt_status_message}")
                else:
                    st.session_state.stt_microphone_stream_obj = mic_stream
                    audio_generator = mic_stream.generator()
                    stt_config_obj = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=STT_RATE, language_code="en-US",
                        enable_automatic_punctuation=True,
                        # model="chirp", # Consider using chirp for potentially better accuracy if available
                        # use_enhanced=True, # If using a premium model and feature
                    )
                    streaming_config = speech.StreamingRecognitionConfig(
                        config=stt_config_obj,
                        interim_results=True,
                        # single_utterance=False # Set to True if you only want one utterance
                    )
                    st.session_state.stt_audio_thread = threading.Thread(
                        name="STTProcessingThread",
                        target=stt_engine.process_audio_stream,
                        args=(st.session_state.stt_google_client, streaming_config, audio_generator,
                              st.session_state.stt_stop_event_thread, st.session_state.stt_ui_update_queue),
                        daemon=True
                    )
                    st.session_state.stt_audio_thread.start()
                    st.session_state.stt_status_message = "Recording..."
                    app_logger.info("STT audio thread started.")
                st.rerun()

        with col_stop_rec:
            if st.button("⏹️ Stop Recording", disabled=not st.session_state.stt_is_recording, key="stop_rec_btn", use_container_width=True):
                app_logger.info("Stop Recording button clicked.")
                st.session_state.stt_status_message = "Stopping recording... Signaling microphone and STT to stop."
                st.session_state.stt_stop_event_thread.set() # Signal for STT processing loop

                if st.session_state.stt_microphone_stream_obj:
                    app_logger.info("Stop button: Closing microphone stream.")
                    st.session_state.stt_microphone_stream_obj.close_stream()
                    # stt_microphone_stream_obj will be fully None'd in thread_done handler
                else:
                    app_logger.warning("Stop button: stt_microphone_stream_obj is None at stop click.")
                # No st.rerun() here; let queue processor handle it.

        st.info(f"STT Status: {st.session_state.stt_status_message}")
        if st.session_state.stt_error:
            st.error(f"STT Error: {st.session_state.stt_error}")

        st.markdown(f"**Live Transcript:** `{st.session_state.stt_interim_transcript}`")

        st.markdown("#### ⌨️ Or type/edit your query")
        st.text_area(
            "Your Question:",
            key="active_query_text",
            height=100,
            disabled=not st.session_state.document_processed or st.session_state.stt_is_recording
        )

        if st.button("✉️ Ask Gemini", key="ask_rag_btn",
                      disabled=not st.session_state.document_processed or \
                               not st.session_state.get("active_query_text", "").strip() or \
                               st.session_state.stt_is_recording, use_container_width=True):
            st.session_state.submit_rag_query_now = True
            st.rerun()

    st.divider()

    # --- Section 3: Conversation History (Assuming no changes needed here) ---
    st.header("📜 Conversation History")
    if st.button("🗑️ Clear Chat History", key="clear_chat_hist_btn_integrated",
                  disabled=not st.session_state.rag_chat_history, use_container_width=True):
        st.session_state.rag_chat_history = []
        save_rag_chat_history()
        st.session_state.debug_retrieved_chunks = []
        st.session_state.all_processed_chunks_for_debug = []
        st.session_state.active_query_text = ""
        st.success("Chat history cleared.")
        st.rerun()

    if st.session_state.rag_chat_history:
        for i in range(len(st.session_state.rag_chat_history) - 1, -1, -1):
            turn = st.session_state.rag_chat_history[i]
            with st.chat_message("user"):
                st.markdown(turn['question'])
            with st.chat_message("assistant"):
                st.markdown(turn['answer'], help="Error" if turn.get('is_error') else None)
    elif st.session_state.document_processed:
        st.caption("No questions asked yet for the current document.")


    # --- STT Queue Processing ---
    ui_updated_by_queue = False
    while not st.session_state.stt_ui_update_queue.empty():
        try:
            msg = st.session_state.stt_ui_update_queue.get_nowait()
            ui_updated_by_queue = True
            msg_type = msg.get("type")
            app_logger.debug(f"Processing STT UI queue message: {msg_type}")


            if msg_type == "interim":
                st.session_state.stt_interim_transcript = msg["text"]
            elif msg_type == "final":
                st.session_state.stt_full_transcript += msg["text"] + " "
                st.session_state.stt_interim_transcript = ""
            elif msg_type == "status":
                st.session_state.stt_status_message = msg["message"]
                app_logger.info(f"STT Status update: {msg['message']}")
                # If STT thread signals stream ended (e.g., due to limit)
                if "Stream limit" in msg["message"] or "Stream ended" in msg["message"]:
                    if st.session_state.stt_is_recording: # Ensure we only try to stop if it thinks it's recording
                       app_logger.info("STT indicated stream end, ensuring stop event is set.")
                       st.session_state.stt_stop_event_thread.set()
                       # Mic stream might also need explicit closing if not done by STT thread's natural end
                       if st.session_state.stt_microphone_stream_obj:
                           st.session_state.stt_microphone_stream_obj.close_stream()
            elif msg_type == "error":
                st.session_state.stt_error = msg["message"]
                st.session_state.stt_status_message = f"STT Error: {msg['message']}"
                app_logger.error(f"STT Error from queue: {msg['message']}")
                if st.session_state.stt_is_recording:
                    app_logger.info("STT error occurred, ensuring stop event is set.")
                    st.session_state.stt_stop_event_thread.set()
                    if st.session_state.stt_microphone_stream_obj:
                        st.session_state.stt_microphone_stream_obj.close_stream()
            elif msg_type == "thread_done":
                app_logger.info("STT 'thread_done' message received. Finalizing STT session.")
                st.session_state.stt_is_recording = False

                if st.session_state.stt_audio_thread and st.session_state.stt_audio_thread.is_alive():
                    app_logger.info("Thread_done: Joining STT audio thread.")
                    st.session_state.stt_audio_thread.join(timeout=5)
                    if st.session_state.stt_audio_thread.is_alive():
                         app_logger.warning("Thread_done: STT audio thread did not join in time.")
                st.session_state.stt_audio_thread = None

                # Ensure microphone stream is fully closed and dereferenced.
                # Should have been initiated by stop button or error, this is a final cleanup.
                if st.session_state.stt_microphone_stream_obj:
                    app_logger.info("Thread_done: Ensuring microphone stream object is fully closed and None.")
                    st.session_state.stt_microphone_stream_obj.close_stream()
                    st.session_state.stt_microphone_stream_obj = None

                final_transcript = st.session_state.stt_full_transcript.strip()
                if final_transcript:
                    st.session_state.stt_transcript_to_populate = final_transcript
                    st.session_state.stt_status_message = "Recording finished. Transcript populated."
                    app_logger.info(f"Final transcript: '{final_transcript}'")
                else:
                    st.session_state.stt_status_message = "Recording finished. No transcript captured."
                    app_logger.info("No final transcript captured.")

                st.session_state.stt_full_transcript = ""
                st.session_state.stt_interim_transcript = ""

        except queue.Empty:
            break
        except Exception as e:
            app_logger.error(f"Error processing STT UI queue: {e}", exc_info=True)
            st.session_state.stt_error = f"App queue processing error: {e}"
            st.session_state.stt_status_message = "Error processing voice input."
            ui_updated_by_queue = True
            if st.session_state.stt_is_recording: # Attempt to stop if something went wrong
                st.session_state.stt_stop_event_thread.set()
                if st.session_state.stt_microphone_stream_obj:
                    st.session_state.stt_microphone_stream_obj.close_stream()


    if ui_updated_by_queue:
        st.rerun()


    # --- RAG Query Processing (Assuming no changes needed here from your previous version) ---
    if st.session_state.submit_rag_query_now and st.session_state.get("active_query_text", "").strip():
        question_to_process = st.session_state.active_query_text.strip()
        st.session_state.submit_rag_query_now = False

        with st.spinner("🤖 Thinking..."):
            target_user_language_code = rag_engine.LANGUAGE_OPTIONS_MAP.get(st.session_state.current_language_display, "en")
            target_user_language_name = st.session_state.current_language_display
            app_logger.info(f"Processing RAG query in {target_user_language_name}: '{question_to_process}'")

            final_answer = "Error: Could not process RAG query."
            is_error_flag = True
            st.session_state.debug_retrieved_chunks = []

            if rag_engine.global_vector_store.index.ntotal == 0:
                final_answer = "No document context available. Please process a document first."
            else:
                question_for_search = question_to_process
                if target_user_language_code != rag_engine.PDF_CONTENT_LANGUAGE_CODE:
                    translated_q = rag_engine.translate_text_gemini(
                        question_to_process,
                        rag_engine.PDF_CONTENT_LANGUAGE_CODE,
                        source_language_code=target_user_language_code
                    )
                    if translated_q:
                        question_for_search = translated_q
                        app_logger.info(f"Translated query for search: '{question_for_search}'")
                    else:
                        app_logger.warning("Translation of query failed, using original query for search.")

                query_emb = rag_engine.get_google_embedding(question_for_search, task_type="RETRIEVAL_QUERY")
                if query_emb:
                    context_chunks = rag_engine.global_vector_store.search(query_emb, top_k=5)
                    st.session_state.debug_retrieved_chunks = context_chunks

                    if context_chunks:
                        context_str = "\n\n---\n\n".join(context_chunks)
                        prompt_to_llm = f"""You are an AI assistant. Answer the user's question based *only* on the provided context.
User question (in {target_user_language_name}): "{question_to_process}"
Context from document (in {rag_engine.PDF_CONTENT_LANGUAGE_DISPLAY}):
---
{context_str}
---
If the context does not contain the answer, state that the information is not available in the provided excerpts.
Your entire answer must be in {target_user_language_name}.

Answer:"""
                        try:
                            safety_settings_gemini = [
                                {"category": genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                                {"category": genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                                {"category": genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                                {"category": genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
                            ]
                            response = rag_engine.gemini_chat_model.generate_content(
                                prompt_to_llm,
                                generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=8192),
                                safety_settings=safety_settings_gemini
                            )
                            if response.candidates and response.text:
                                final_answer = response.text.strip()
                                is_error_flag = False
                            else:
                                final_answer = f"AI response was blocked or empty. Prompt feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}"
                                app_logger.warning(f"Gemini response issue: {final_answer}")
                        except Exception as e:
                            app_logger.error(f"Gemini error during RAG: {e}", exc_info=True)
                            final_answer = f"AI error during RAG generation: {str(e)[:200]}..."
                    else:
                        final_answer = "No relevant information found in the document for your query."
                else:
                    final_answer = "Could not create embedding for the query."

            st.session_state.rag_chat_history.append({
                'question': question_to_process, 'answer': final_answer,
                'is_error': is_error_flag, 'timestamp': datetime.now().isoformat()
            })
            save_rag_chat_history()
            st.session_state.clear_query_box_on_next_run = True
            st.rerun()

    # Continuous UI update for interim transcript if recording
    if st.session_state.stt_is_recording and not ui_updated_by_queue:
        # This helps keep the interim transcript responsive but can cause many reruns.
        # Ensure the sleep time is not too small to avoid excessive CPU.
        time.sleep(0.15) # Slightly increased from 0.1
        st.rerun()

    if DEBUG_MODE:
        st.divider()
        st.subheader("⚙️ Developer Debug Information")
        if st.session_state.all_processed_chunks_for_debug:
            with st.expander("DEV_DEBUG: ALL Processed Chunks from Document", expanded=False):
                st.write(f"Total chunks created: {len(st.session_state.all_processed_chunks_for_debug)}")

        if st.session_state.debug_retrieved_chunks:
            with st.expander("DEV_DEBUG: Last Retrieved Chunks for RAG", expanded=False):
                st.write(f"Total chunks retrieved for last query: {len(st.session_state.debug_retrieved_chunks)}")

if __name__ == "__main__":
    main()