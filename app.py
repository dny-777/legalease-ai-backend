import uvicorn
import os
import uuid
import json
import logging
import mimetypes
import requests # <-- We use this for API calls now
from typing import Optional # Import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# OCR
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io

# RAG
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Audio
from gtts import gTTS

# Your API Schemas
from schemas import *

# --- 1. INITIAL SETUP & CONFIG ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CREATE STATIC DIRS **BEFORE** APP MOUNT --- MOVED HERE ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("audio", exist_ok=True)
# -----------------------------------------------------------

# --- NEW: Hugging Face API Config ---
HF_API_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
HF_API_BASE_URL = "https://api-inference.huggingface.co/models/"

# --- 2. GLOBAL VARIABLES & HELPERS ---
models = {} # Will hold our loaded models (OCR, RAG)

# --- UPDATED LANG_CODES ---
# gTTS codes AND mBART codes
# Format: "LanguageName": {"gtts": "gtts_code", "mbart": "mbart_code"}
LANG_CODES = {
    "English":   {"gtts": "en", "mbart": "en_XX"},
    "Hindi":     {"gtts": "hi", "mbart": "hi_IN"},
    "Marathi":   {"gtts": "mr", "mbart": "mr_IN"},
    "Tamil":     {"gtts": "ta", "mbart": "ta_IN"},
    "Telugu":    {"gtts": "te", "mbart": "te_IN"},
    "Bengali":   {"gtts": "bn", "mbart": "bn_IN"},
    "Gujarati":  {"gtts": "gu", "mbart": "gu_IN"},
    "Kannada":   {"gtts": "kn", "mbart": "kn_IN"}, # mBART might not support Kannada well
    "Malayalam": {"gtts": "ml", "mbart": "ml_IN"},
    "Punjabi":   {"gtts": "pa", "mbart": "pa_IN"}  # mBART might not support Punjabi well
    # Add more languages if mBART supports them: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages
}

def init_json_db(filepath: str, default_data):
    if not os.path.exists(filepath):
        logger.warning(f"Creating empty file: {filepath}")
        with open(filepath, "w") as f:
            json.dump(default_data, f, indent=4)

def save_upload_data(file_id, original_name, file_path, raw_text):
    try:
        # Ensure data directory exists before trying to open the file
        os.makedirs("data", exist_ok=True)
        # Use 'a+' mode initially to create if not exists, then read/write
        try:
            with open("data/uploads.json", "r+") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, dict): # Handle case where file exists but isn't a dict
                         data = {}
                except json.JSONDecodeError:
                    data = {} # Start fresh if file is corrupt/empty
                data[file_id] = {
                    "original_name": original_name, "file_path": file_path,
                    "raw_text": raw_text, "simplified_text": ""
                }
                f.seek(0)
                f.truncate() # Clear before writing
                json.dump(data, f, indent=4)
        except FileNotFoundError: # Should ideally not happen with 'a+' but as fallback
             with open("data/uploads.json", "w") as f:
                 data = {}
                 data[file_id] = {
                    "original_name": original_name, "file_path": file_path,
                    "raw_text": raw_text, "simplified_text": ""
                 }
                 json.dump(data, f, indent=4)

    except Exception as e:
        logger.error(f"Failed to save upload data: {e}")


# --- CORRECTED TTS HELPER ---
def generate_tts(text: str, gtts_lang_code: str) -> Optional[str]:
    if not gtts_lang_code:
        logger.warning(f"No gTTS lang code provided, skipping TTS.")
        return None
    try:
        tts = gTTS(text=text, lang=gtts_lang_code, slow=False, tld='co.in')
        file_id = str(uuid.uuid4())
        # Ensure audio directory exists
        os.makedirs("audio", exist_ok=True)
        relative_file_path = f"audio/{file_id}.mp3"
        tts.save(relative_file_path)
        return f"/{relative_file_path}"
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        return None

# --- NEW: Hugging Face API Helper Function ---
def hf_api_query(payload, model_url):
    """Generic function to query the HF Inference API."""
    if not HF_API_TOKEN:
         logger.error("Hugging Face API token (HF_TOKEN) is not set.")
         raise HTTPException(status_code=500, detail="Server configuration error: Hugging Face API token not set.")
    try:
        response = requests.post(model_url, headers=HF_HEADERS, json=payload, timeout=60) # Added timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to Hugging Face API: {model_url}")
        raise HTTPException(status_code=504, detail="Hugging Face API request timed out.")
    except requests.exceptions.RequestException as errh: # Broader exception for network issues
        logger.error(f"Hugging Face API request error: {errh} for URL {model_url}")
        detail_msg = str(errh)
        status_code = 500 # Default internal server error
        # Try to get specific details if it was an HTTP error
        if hasattr(errh, 'response') and errh.response is not None:
             status_code = errh.response.status_code
             detail_msg = errh.response.text
             try:
                 detail_msg = errh.response.json()
             except json.JSONDecodeError:
                 pass
             if status_code == 503 and "loading" in errh.response.text.lower():
                  raise HTTPException(status_code=503, detail=f"Model is loading, please try again. Error: {detail_msg}")
        raise HTTPException(status_code=status_code, detail=f"Hugging Face API error: {detail_msg}")
    except Exception as e: # Catch other potential errors
        logger.error(f"Unexpected error during Hugging Face query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query Hugging Face: {str(e)}")


# --- 3. STARTUP (LIFESPAN) EVENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")

    # 1. Create dummy data/kb files & folders (uploads/audio moved outside)
    os.makedirs("data", exist_ok=True)
    os.makedirs("kb", exist_ok=True)
    init_json_db("data/uploads.json", {})
    init_json_db("data/cases.json", [])
    if not os.path.exists("kb/kb.json"):
        init_json_db("kb/kb.json", [{"id": "dummy", "text": "This is a dummy knowledge base."}])

    # 2. Load EasyOCR (local, free)
    logger.info("Loading EasyOCR model...")
    try:
        models['ocr_reader'] = easyocr.Reader(['en', 'hi', 'mr'], gpu=False) # Specify CPU explicitly
        logger.info("EasyOCR model loaded successfully.")
    except Exception as e:
         logger.error(f"Failed to load EasyOCR model: {e}", exc_info=True)
         # Optionally raise error or continue without OCR

    # 3. Load RAG models (local, free)
    try:
        if not os.path.exists("kb/index.faiss"):
            logger.warning("FAISS index not found. Running build_kb...")
            # Ensure build_kb exists and handles its own errors
            try:
                from build_kb import build_kb
                build_kb()
            except ImportError:
                 logger.error("build_kb.py not found or cannot be imported.")
            except Exception as build_e:
                 logger.error(f"Error running build_kb: {build_e}", exc_info=True)

        # Check again if index exists before trying to load
        if os.path.exists("kb/index.faiss") and os.path.exists("kb/index_mapping.json") and os.path.exists("kb/kb.json"):
            logger.info("Loading SentenceTransformer model...")
            models['st_model'] = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loading FAISS index...")
            models['faiss_index'] = faiss.read_index("kb/index.faiss")

            with open("kb/kb.json", "r", encoding="utf-8") as f:
                models['kb_data_map'] = {item['id']: item['text'] for item in json.load(f)}
            with open("kb/index_mapping.json", "r") as f:
                models['index_mapping'] = {int(k): v for k, v in json.load(f).items()}

            logger.info("All RAG models loaded successfully.")
        else:
             logger.error("One or more RAG files (index, mapping, kb.json) are missing after build attempt. Chat functionality might be limited.")

    except Exception as e:
        logger.error(f"Failed to load RAG models during startup: {e}", exc_info=True)
        # Decide if the app should fail completely or continue with limited functionality

    yield # App runs here

    # --- Shutdown ---
    logger.info("Application shutdown...")
    models.clear() # Clear models from memory

# --- 4. FASTAPI APP INITIALIZATION ---
# Initialize AFTER creating directories needed for mounting
app = FastAPI(
    title="LegalEase AI API (100% Free)",
    description="Full-stack API for multilingual legal assistance using Hugging Face.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan context manager
)

# --- Mount static directories AFTER app initialization ---
# Ensure these directories exist before mounting
try:
    app.mount("/audio", StaticFiles(directory="audio"), name="audio")
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
except RuntimeError as mount_error:
     logger.error(f"Failed to mount static directories: {mount_error}. Ensure 'audio' and 'uploads' folders exist.")
     # Decide if this is a fatal error for your application


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity in hackathon
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)


# --- 5. API ENDPOINTS (NOW 100% FREE) ---

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the LegalEase AI API (100% Free)!"}

# === ROUTE 1: OCR ===
@app.post("/api/upload_ocr", response_model=UploadOCRResponse, tags=["OCR"])
async def upload_and_ocr(file: UploadFile = File(...)):
    reader = models.get('ocr_reader')
    if not reader:
        raise HTTPException(status_code=503, detail="OCR service is not ready or failed to load.")

    file_id = str(uuid.uuid4())
    # Sanitize filename slightly - remove potential path traversal chars
    safe_filename = os.path.basename(file.filename or f"{file_id}_upload")
    file_extension = safe_filename.split('.')[-1].lower() if '.' in safe_filename else ''
    file_path = f"uploads/{file_id}.{file_extension}"

    # Ensure uploads directory exists
    os.makedirs("uploads", exist_ok=True)

    contents = await file.read()
    try:
        with open(file_path, "wb") as f:
            f.write(contents)
    except IOError as e:
         logger.error(f"Failed to save uploaded file {file_path}: {e}")
         raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    raw_text = ""
    try:
        if file_extension == 'pdf':
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png") # Ensure format is compatible with easyocr
                # Use easyocr directly on bytes
                result = reader.readtext(img_bytes)
                page_text = ' '.join([text[1] for text in result])
                raw_text += page_text + "\n\n"
            doc.close()
        elif file_extension in ['png', 'jpg', 'jpeg']:
            # Read directly from saved file path
            result = reader.readtext(file_path)
            raw_text = ' '.join([text[1] for text in result])
        else:
            # Clean up the invalid file before raising error
            try: os.remove(file_path)
            except OSError: pass
            raise HTTPException(status_code=400, detail=f"Unsupported file type: .{file_extension}")

        raw_text = ' '.join(raw_text.split()) # Clean multiple whitespaces
        save_upload_data(file_id, safe_filename, file_path, raw_text)
        return UploadOCRResponse(file_id=file_id, file_name=safe_filename, raw_text=raw_text)

    except Exception as e:
        logger.error(f"OCR processing failed for {file_path}: {e}", exc_info=True)
        # Clean up the file if processing failed
        try: os.remove(file_path)
        except OSError: pass
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

# === ROUTE 2: SIMPLIFY (FREE) ===
@app.post("/api/simplify", response_model=SimplifyResponse, tags=["NLP"])
async def simplify_text(request: TextRequest):
    payload = {"inputs": f"summarize: {request.text}"}
    model_url = f"{HF_API_BASE_URL}sshleifer/distilbart-cnn-12-6"
    response_data = hf_api_query(payload, model_url)
    if isinstance(response_data, list) and len(response_data) > 0:
        simplified = response_data[0].get('summary_text')
        if simplified is not None:
             return SimplifyResponse(simplified_text=simplified)
    logger.error(f"Unexpected API response format for simplify: {response_data}")
    raise HTTPException(status_code=500, detail="API response format invalid or empty.")

# === ROUTE 3: TRANSLATE (FREE - Using mBART) --- UPDATED --- ===
@app.post("/api/translate", response_model=TranslateResponse, tags=["NLP"])
async def translate_text(request: LanguageRequest):
    lang_details = LANG_CODES.get(request.target_language)
    if not lang_details or not lang_details.get("mbart"):
         raise HTTPException(status_code=400, detail=f"Language '{request.target_language}' not supported for mBART translation.")
    mbart_lang_code = lang_details["mbart"]
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model_url = f"{HF_API_BASE_URL}{model_name}"
    payload = { "inputs": request.text, "parameters": { "src_lang": "en_XX", "tgt_lang": mbart_lang_code } }
    logger.info(f"Calling mBART for {request.target_language} ({mbart_lang_code})")
    response_data = hf_api_query(payload, model_url)
    if isinstance(response_data, list) and len(response_data) > 0:
        translated = response_data[0].get('translation_text')
        if translated is not None:
             return TranslateResponse(translated_text=translated)
    logger.error(f"Unexpected API response format for translate: {response_data}")
    raise HTTPException(status_code=500, detail="Translation API response format invalid or empty.")

# === ROUTE 4: TEXT-TO-SPEECH (TTS) --- UPDATED gTTS code lookup --- ===
@app.post("/api/tts", response_model=TTSResponse, tags=["Audio"])
async def text_to_speech(request: TTSRequest, http_request: Request):
    lang_details = LANG_CODES.get(request.language)
    gtts_lang_code = lang_details.get("gtts") if lang_details else None
    if not gtts_lang_code:
         raise HTTPException(status_code=400, detail=f"Language '{request.language}' not supported for TTS.")
    relative_path = generate_tts(request.text, gtts_lang_code)
    if not relative_path:
        raise HTTPException(status_code=500, detail="TTS generation failed.")
    base_url = str(http_request.base_url).rstrip('/')
    audio_url = f"{base_url}{relative_path}"
    return TTSResponse(audio_url=audio_url)

# === ROUTE 5: SPEECH-TO-TEXT (STT) (FREE) --- DISABLED DUE TO API UNRELIABILITY --- ===
# @app.post("/api/stt", response_model=STTResponse, tags=["Audio"])
# async def speech_to_text(file: UploadFile = File(...)):
#     # ... (code remains commented out) ...
#     pass # Add pass to make the commented function valid Python syntax if uncommented partially


# === ROUTE 6: CHAT (THE BRAIN) (FREE) --- UPDATED --- ===
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, http_request: Request):
    st_model = models.get('st_model')
    index = models.get('faiss_index')
    mapping = models.get('index_mapping')
    kb_data = models.get('kb_data_map')

    if not all([st_model, index, mapping, kb_data]):
        logger.error("RAG models not loaded correctly. Cannot process chat request.")
        raise HTTPException(status_code=503, detail="Chat service (RAG models) is not ready.")

    try:
        # --- 1. RAG: Get Knowledge Base Context ---
        query_embedding = st_model.encode([request.query])
        k = 1
        distances, indices = index.search(query_embedding, k)
        best_match_text = "I'm sorry, I don't have specific information on that in my knowledge base."
        if indices.size > 0 and indices[0][0] != -1:
            faiss_index = indices[0][0]
            original_id = mapping.get(faiss_index)
            if original_id:
                best_match_text = kb_data.get(original_id, best_match_text)
            else:
                 logger.warning(f"FAISS index {faiss_index} not found in mapping.")
        else:
             logger.info(f"No relevant KB entry found for query: {request.query}")

        # --- 2. Build Answer (Translate if needed, using mBART) ---
        answer_text = best_match_text
        if request.language != "English":
            lang_details = LANG_CODES.get(request.language)
            if lang_details and lang_details.get("mbart"):
                mbart_lang_code = lang_details["mbart"]
                trans_model_name = "facebook/mbart-large-50-many-to-many-mmt"
                trans_model_url = f"{HF_API_BASE_URL}{trans_model_name}"
                trans_payload = { "inputs": best_match_text, "parameters": { "src_lang": "en_XX", "tgt_lang": mbart_lang_code } }
                try:
                    logger.info(f"Translating chat answer to {request.language} ({mbart_lang_code}) using mBART...")
                    trans_response = hf_api_query(trans_payload, trans_model_url)
                    if isinstance(trans_response, list) and len(trans_response) > 0:
                        translated_answer = trans_response[0].get('translation_text')
                        if translated_answer:
                            answer_text = translated_answer
                        else:
                            logger.warning(f"mBART translation succeeded but response empty. Using English.")
                    else:
                        logger.warning(f"Unexpected mBART translation response. Using English.")
                except Exception as e:
                    logger.warning(f"mBART Translation failed: {e}. Using English answer.")
            else:
                 logger.warning(f"Language {request.language} not supported for mBART. Using English.")
        else:
             logger.info("Chat language is English, skipping translation.")

        # --- 3. Generate Audio for Answer ---
        gtts_lang_code = LANG_CODES.get(request.language, {}).get("gtts", "en")
        relative_audio_path = generate_tts(answer_text, gtts_lang_code)
        audio_url = None
        if relative_audio_path:
            base_url = str(http_request.base_url).rstrip('/')
            audio_url = f"{base_url}{relative_audio_path}"

        # --- 4. Save Chat & Return ---
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            # Use 'a+' mode initially, then read/write carefully
            try:
                with open("data/cases.json", "r+") as f:
                    try:
                        history = json.load(f)
                        if not isinstance(history, list): history = []
                    except json.JSONDecodeError:
                        history = []
                    history.append({ "query": request.query, "answer": answer_text, "file_id": request.file_id, "audio_url": audio_url })
                    f.seek(0)
                    f.truncate()
                    json.dump(history, f, indent=4)
            except FileNotFoundError:
                 with open("data/cases.json", "w") as f:
                     history = [{ "query": request.query, "answer": answer_text, "file_id": request.file_id, "audio_url": audio_url }]
                     json.dump(history, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}", exc_info=True) # Log full traceback

        return ChatResponse(
            answer_text=answer_text, audio_url=audio_url,
            file_id=request.file_id, query=request.query
        )

    except Exception as e:
        logger.error(f"Chat processing failed unexpectedly: {e}", exc_info=True)
        detail = f"Chat processing failed: {str(e)}"
        if isinstance(e, HTTPException): detail = e.detail
        raise HTTPException(status_code=500, detail=detail)

import os
import uvicorn
import mimetypes

if __name__ == "__main__":
    print("✅ Starting backend...")

    # Fix for MIME types (optional but harmless)
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    # Ensure Render port binding
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Running on port {port}")

    # Launch Uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port)

