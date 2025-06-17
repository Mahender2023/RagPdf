# rag_engine.py
import fitz  # PyMuPDF
import google.generativeai as genai
import faiss
import numpy as np
import logging
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx
from PIL import Image
import io

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini Models and Configuration ---
gemini_chat_model = None
google_embedding_model_name = "models/embedding-001"
EMBEDDING_DIMENSION = 768
PDF_CONTENT_LANGUAGE_CODE = "en"
PDF_CONTENT_LANGUAGE_DISPLAY = "English"

LANGUAGE_OPTIONS_MAP = {
    "English": "en", "中文 (Chinese Simplified)": "zh-CN", "Español (Spanish)": "es",
    "Русский (Russian)": "ru", "తెలుగు (Telugu)": "te", "हिन्दी (Hindi)": "hi",
    "ಕನ್ನಡ (Kannada)": "kn", "தமிழ் (Tamil)": "ta"
}
LANGUAGE_CODE_TO_DISPLAY_MAP = {v: k for k, v in LANGUAGE_OPTIONS_MAP.items()}

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_chat_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logging.info("Google AI client (rag_engine.py) configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Google AI client in rag_engine.py: {e}")
        gemini_chat_model = None
else:
    logging.warning("GOOGLE_API_KEY not found in rag_engine.py. Google AI features will be disabled if not set by main app.")
    gemini_chat_model = None

# --- Document Parsers ---
def load_pdf_pages_pymupdf(pdf_file_stream):
    pages_text_list = []
    try:
        doc = fitz.open(stream=pdf_file_stream.read(), filetype="pdf")
        if doc.is_encrypted:
            logging.warning("Uploaded PDF is encrypted. Attempting to authenticate with empty password.")
            if not doc.authenticate(""):
                logging.error("PDF decryption failed. Cannot process.")
                doc.close()
                return []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text and text.strip():
                pages_text_list.append(text.strip())
            else:
                logging.info(f"Page {page_num + 1} in PDF has no extractable text.")
        doc.close()
        if not pages_text_list:
            logging.warning("No text extracted from PDF.")
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
    return pages_text_list

def load_docx_text(file_stream):
    full_text_content = ""
    try:
        doc = docx.Document(file_stream)
        for para in doc.paragraphs:
            full_text_content += para.text + "\n"
        logging.info("Extracted text from DOCX.")
    except Exception as e:
        logging.error(f"Error processing DOCX: {e}")
    return full_text_content.strip()

def get_image_description_gemini(image_file_stream, image_filename="image"):
    if not gemini_chat_model:
        logging.error("Gemini chat model not initialized. Cannot get image description.")
        return None
    try:
        img_bytes = image_file_stream.read()
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        prompt = f"Describe this image ({image_filename}) in detail for Retrieval Augmented Generation. Focus on objects, actions, setting, and any visible text. Be comprehensive and objective."
        response = gemini_chat_model.generate_content([prompt, img])
        
        if response.text:
            logging.info(f"Generated description for image '{image_filename}'.")
            return response.text.strip()
        
        logging.warning(f"No description generated for image '{image_filename}'. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        return None
    except Exception as e:
        logging.error(f"Error generating image description for '{image_filename}': {e}")
        return None

# --- Text Chunking ---
def get_text_chunks(text_content, chunk_size=1800, chunk_overlap=250):
    if not text_content or not text_content.strip():
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ",", " ", ""],
        keep_separator=False
    )
    chunks = text_splitter.split_text(text_content)
    meaningful_chunks = [chk.strip() for chk in chunks if len(chk.strip()) > 50]
    if not meaningful_chunks and chunks:
        return [chk.strip() for chk in chunks if chk.strip()]
    return meaningful_chunks

# --- Embedding Function ---
def get_google_embedding(text_to_embed, task_type="RETRIEVAL_DOCUMENT"):
    if not gemini_chat_model:
        logging.error("Google AI not configured. Cannot get embedding.")
        return None
    if not text_to_embed or not text_to_embed.strip():
        logging.warning("Attempted to embed empty or whitespace-only text.")
        return None
    try:
        if not isinstance(text_to_embed, str):
            text_to_embed = str(text_to_embed)

        response = genai.embed_content(
            model=google_embedding_model_name,
            content=text_to_embed,
            task_type=task_type
        )
        return response['embedding']
    except Exception as e:
        logging.error(f"Error getting Google embedding for '{text_to_embed[:50]}...': {e}")
        return None

# --- Translation Function ---
def translate_text_gemini(text_to_translate, target_language_code, source_language_code="auto"):
    if not gemini_chat_model:
        logging.error("Gemini chat model not initialized. Cannot translate.")
        return None
    if not text_to_translate or not text_to_translate.strip():
        return text_to_translate

    try:
        target_language_name = LANGUAGE_CODE_TO_DISPLAY_MAP.get(target_language_code, target_language_code)
        source_language_info = "automatically detected language"
        if source_language_code != "auto":
            source_language_info = LANGUAGE_CODE_TO_DISPLAY_MAP.get(source_language_code, source_language_code)

        prompt = f"Translate the following text from {source_language_info} to {target_language_name}. Only provide the translated text, no other commentary or explanation.\n\nText to translate: \"{text_to_translate}\""
        
        response = gemini_chat_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        if response.candidates and response.text:
            return response.text.strip()
        
        logging.warning(f"Translation failed for '{text_to_translate[:50]}...'. Feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
        return text_to_translate
    except Exception as e:
        logging.error(f"Error during translation of '{text_to_translate[:50]}...': {e}")
        return text_to_translate

# --- Vector Store Class (Singleton Pattern) ---
class VectorStore:
    _instance = None

    def __new__(cls, *args, **kwargs): # *args, **kwargs are fine here to accept arguments
        if not cls._instance:
            # CORRECTED LINE: Do not pass *args, **kwargs to object.__new__
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialized = False # Initialize this flag on the new instance
        return cls._instance

    def __init__(self, embedding_dim=EMBEDDING_DIMENSION):
        # This check ensures that __init__ logic runs only once for the singleton
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.embedding_dim = embedding_dim # Use the passed argument
        self.reset() # Initializes index and texts
        self._initialized = True # Mark as initialized
        logging.info(f"VectorStore singleton instance initialized with dimension {self.embedding_dim}.")

    def reset(self):
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.texts = []
        logging.info(f"VectorStore reset. Index is new, {self.index.ntotal} items.")

    def add(self, text_chunk, embedding_vector):
        if embedding_vector is None:
            logging.warning(f"Skipping add to VectorStore: embedding is None for text chunk '{text_chunk[:30]}...'.")
            return
        if not isinstance(embedding_vector, list) or len(embedding_vector) != self.embedding_dim:
            logging.error(f"Invalid embedding vector for '{text_chunk[:30]}...'. Expected list of dim {self.embedding_dim}, got {type(embedding_vector)} of len {len(embedding_vector) if isinstance(embedding_vector, list) else 'N/A'}.")
            return
        try:
            self.index.add(np.array([embedding_vector], dtype='float32'))
            self.texts.append(text_chunk)
        except Exception as e:
            logging.error(f"Error adding to FAISS index for '{text_chunk[:30]}...': {e}")

    def search(self, query_embedding_vector, top_k=10):
        if self.index.ntotal == 0:
            logging.warning("Search called on an empty VectorStore index.")
            return []
        if query_embedding_vector is None:
            logging.warning("Search called with None query_embedding_vector.")
            return []
        if not isinstance(query_embedding_vector, list) or len(query_embedding_vector) != self.embedding_dim:
            logging.error(f"Invalid query embedding vector. Expected list of dim {self.embedding_dim}.")
            return []
        
        try:
            actual_top_k = min(top_k, self.index.ntotal)
            if actual_top_k == 0: return []

            distances, indices = self.index.search(np.array([query_embedding_vector], dtype='float32'), actual_top_k)
            results = [self.texts[i] for i in indices[0] if 0 <= i < len(self.texts)]
            return results
        except Exception as e:
            logging.error(f"Error searching FAISS index: {e}")
            return []

# Global instance of the VectorStore
# This call will now work correctly: __new__ accepts embedding_dim via kwargs,
# but doesn't pass it to super().__new__(). __init__ will receive it.
global_vector_store = VectorStore(embedding_dim=EMBEDDING_DIMENSION)

if not gemini_chat_model and GOOGLE_API_KEY:
    try:
        gemini_chat_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logging.info("Re-checked and configured Gemini chat model in rag_engine.")
    except Exception as e:
        logging.error(f"Failed to re-check and configure Gemini chat model: {e}")