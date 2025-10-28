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
        with open("data/uploads.json", "r+") as f:
            data = json.load(f)
            data[file_id] = {
                "original_name": original_name,
                "file_path": file_path,
                "raw_text": raw_text,
                "simplified_text": ""
            }
            f.seek(0)
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save upload data: {e}")

# --- CORRECTED TTS HELPER ---
# Takes the gTTS language code (e.g., 'hi')
# Returns the relative path like '/audio/uuid.mp3'
def generate_tts(text: str, gtts_lang_code: str) -> Optional[str]:
    if not gtts_lang_code:
        logger.warning(f"No gTTS lang code provided, skipping TTS.")
        return None
    try:
        # Using the tld='co.in' based on previous tests
        tts = gTTS(text=text, lang=gtts_lang_code, slow=False, tld='co.in')
        file_id = str(uuid.uuid4())
        # Store only the relative path part
        relative_file_path = f"audio/{file_id}.mp3"
        # Save using the full path needed by the system
        tts.save(relative_file_path)
        # Return the relative path prefixed with a '/' for URL construction
        return f"/{relative_file_path}"
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        return None

# --- NEW: Hugging Face API Helper Function ---
def hf_api_query(payload, model_url):
    """Generic function to query the HF Inference API."""
    try:
        response = requests.post(model_url, headers=HF_HEADERS, json=payload)
        response.raise_for_status() # Raise an error for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as errh:
        logger.error(f"Http Error: {errh} for URL {model_url}")
        detail_msg = errh.response.text
        try:
            detail_msg = errh.response.json() # Try parsing JSON error
        except json.JSONDecodeError:
            pass # Keep text if not JSON

        if errh.response.status_code == 503 and "loading" in errh.response.text:
            raise HTTPException(status_code=503, detail=f"Model is loading, please try again in a few seconds. Error: {detail_msg}")
        raise HTTPException(status_code=errh.response.status_code, detail=f"Hugging Face API error: {detail_msg}")
    except Exception as e:
        logger.error(f"Hugging Face query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query Hugging Face: {str(e)}")


# --- 3. STARTUP (LIFESPAN) EVENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")

    # 1. Create dummy files & folders
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("audio", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("kb", exist_ok=True)
    init_json_db("data/uploads.json", {})
    init_json_db("data/cases.json", [])
    if not os.path.exists("kb/kb.json"):
        init_json_db("kb/kb.json", [{"id": "dummy", "text": "This is a dummy knowledge base."}])

    # 2. Load EasyOCR (local, free)
    logger.info("Loading EasyOCR model...")
    models['ocr_reader'] = easyocr.Reader(['en', 'hi', 'mr'], gpu=False)
    logger.info("EasyOCR model loaded.")

    # 3. Load RAG models (local, free)
    try:
        if not os.path.exists("kb/index.faiss"):
            logger.warning("FAISS index not found. Running build_kb...")
            from build_kb import build_kb
            build_kb()

        logger.info("Loading SentenceTransformer model...")
        models['st_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loading FAISS index...")
        models['faiss_index'] = faiss.read_index("kb/index.faiss")

        with open("kb/kb.json", "r", encoding="utf-8") as f:
            models['kb_data_map'] = {item['id']: item['text'] for item in json.load(f)}
        with open("kb/index_mapping.json", "r") as f:
            models['index_mapping'] = {int(k): v for k, v in json.load(f).items()}

        logger.info("All RAG models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load RAG models: {e}")

    yield

    # --- Shutdown ---
    logger.info("Application shutdown...")
    models.clear()

# --- 4. FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="LegalEase AI API (100% Free)",
    description="Full-stack API for multilingual legal assistance using Hugging Face.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory="audio"), name="audio")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- 5. API ENDPOINTS (NOW 100% FREE) ---

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the LegalEase AI API (100% Free)!"}

# === ROUTE 1: OCR ===
@app.post("/api/upload_ocr", response_model=UploadOCRResponse, tags=["OCR"])
async def upload_and_ocr(file: UploadFile = File(...)):
    reader = models.get('ocr_reader')
    if not reader:
        raise HTTPException(status_code=503, detail="OCR service is not ready.")

    file_id = str(uuid.uuid4())
    file_extension = file.filename.split('.')[-1].lower()
    file_path = f"uploads/{file_id}.{file_extension}"

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    raw_text = ""
    try:
        if file_extension == 'pdf':
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                result = reader.readtext(img_bytes)
                page_text = ' '.join([text[1] for text in result])
                raw_text += page_text + "\n\n"
            doc.close()
        elif file_extension in ['png', 'jpg', 'jpeg']:
            result = reader.readtext(file_path)
            raw_text = ' '.join([text[1] for text in result])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        raw_text = ' '.join(raw_text.split())
        save_upload_data(file_id, file.filename, file_path, raw_text)
        return UploadOCRResponse(file_id=file_id, file_name=file.filename, raw_text=raw_text)

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

# === ROUTE 2: SIMPLIFY (FREE) ===
@app.post("/api/simplify", response_model=SimplifyResponse, tags=["NLP"])
async def simplify_text(request: TextRequest):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face API token not set.")

    payload = {"inputs": f"summarize: {request.text}"}
    # Using distilbart as t5-small had issues before
    model_url = f"{HF_API_BASE_URL}sshleifer/distilbart-cnn-12-6"

    response_data = hf_api_query(payload, model_url)

    # Check if response is a list and not empty
    if isinstance(response_data, list) and len(response_data) > 0:
        simplified = response_data[0].get('summary_text')
        if simplified is not None:
             return SimplifyResponse(simplified_text=simplified)

    logger.error(f"Unexpected API response format for simplify: {response_data}")
    raise HTTPException(status_code=500, detail="API response format invalid or empty.")


# === ROUTE 3: TRANSLATE (FREE - Using mBART) --- UPDATED --- ===
@app.post("/api/translate", response_model=TranslateResponse, tags=["NLP"])
async def translate_text(request: LanguageRequest):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face API token not set.")

    lang_details = LANG_CODES.get(request.target_language)
    if not lang_details or not lang_details.get("mbart"):
         raise HTTPException(status_code=400, detail=f"Language '{request.target_language}' not supported for mBART translation.")

    mbart_lang_code = lang_details["mbart"] # e.g., "hi_IN", "mr_IN"
    model_name = "facebook/mbart-large-50-many-to-many-mmt" # Multilingual model
    model_url = f"{HF_API_BASE_URL}{model_name}"

    # mBART requires specifying source and target language codes in parameters
    payload = {
        "inputs": request.text,
        "parameters": {
            "src_lang": "en_XX", # Source is English
            "tgt_lang": mbart_lang_code # Target language code from LANG_CODES
        }
    }

    logger.info(f"Calling mBART for {request.target_language} ({mbart_lang_code})")
    response_data = hf_api_query(payload, model_url)

    # Check response format
    if isinstance(response_data, list) and len(response_data) > 0:
        translated = response_data[0].get('translation_text')
        if translated is not None:
             return TranslateResponse(translated_text=translated)

    logger.error(f"Unexpected API response format for translate: {response_data}")
    raise HTTPException(status_code=500, detail="Translation API response format invalid or empty.")


# === ROUTE 4: TEXT-TO-SPEECH (TTS) --- UPDATED gTTS code lookup --- ===
@app.post("/api/tts", response_model=TTSResponse, tags=["Audio"])
async def text_to_speech(request: TTSRequest, http_request: Request):
    # --- UPDATED --- Get gTTS code from the nested dictionary
    lang_details = LANG_CODES.get(request.language)
    gtts_lang_code = lang_details.get("gtts") if lang_details else None

    if not gtts_lang_code:
         raise HTTPException(status_code=400, detail=f"Language '{request.language}' not supported for TTS.")

    # generate_tts now takes gtts_lang_code and returns relative path
    relative_path = generate_tts(request.text, gtts_lang_code)
    if not relative_path:
        # generate_tts logs the error, raise generic error here
        raise HTTPException(status_code=500, detail="TTS generation failed.")

    # Construct the full URL using the server's base URL
    base_url = str(http_request.base_url).rstrip('/')
    audio_url = f"{base_url}{relative_path}" # e.g. "http://127.0.0.1:10000" + "/audio/uuid.mp3"

    return TTSResponse(audio_url=audio_url)

# === ROUTE 5: SPEECH-TO-TEXT (STT) (FREE) --- DISABLED DUE TO API UNRELIABILITY --- ===
# @app.post("/api/stt", response_model=STTResponse, tags=["Audio"])
# async def speech_to_text(file: UploadFile = File(...)):
#     if not HF_API_TOKEN:
#         raise HTTPException(status_code=500, detail="Hugging Face API token not set.")
#
#     # Tried openai/whisper-base, openai/whisper-tiny, facebook/wav2vec2-base-960h
#     # All failed with 404 on free tier during testing.
#     model_url = f"{HF_API_BASE_URL}facebook/wav2vec2-base-960h"
#
#     try:
#         audio_data = await file.read()
#         headers_stt = {"Authorization": f"Bearer {HF_API_TOKEN}"}
#
#         response = requests.post(model_url, headers=headers_stt, data=audio_data)
#         response.raise_for_status()
#         response_data = response.json()
#
#         # Wav2Vec2 might return 'text' or 'error'
#         text = response_data.get("text")
#         error = response_data.get("error")
#
#         if error:
#              logger.error(f"Wav2Vec2 API returned an error: {error}")
#              if isinstance(error, str) and "loading" in error.lower():
#                   raise HTTPException(status_code=503, detail=f"STT model is currently loading, please try again. Error: {error}")
#              raise HTTPException(status_code=500, detail=f"STT API returned an error: {error}")
#
#         if text is None:
#              logger.error(f"STT API response missing 'text' key: {response_data}")
#              raise HTTPException(status_code=500, detail="API response was valid but had no text.")
#
#         transcribed_text = text.strip()
#         return STTResponse(transcribed_text=transcribed_text)
#
#     except requests.exceptions.HTTPError as errh:
#         logger.error(f"Http Error during STT: {errh} for URL {model_url}")
#         detail_msg = errh.response.text
#         try:
#             detail_msg = errh.response.json()
#         except json.JSONDecodeError:
#              pass
#
#         if errh.response.status_code == 503 and "loading" in errh.response.text:
#             raise HTTPException(status_code=503, detail=f"STT model is currently loading, please try again. Error: {detail_msg}")
#         raise HTTPException(status_code=errh.response.status_code, detail=f"Hugging Face STT API error: {detail_msg}")
#     except Exception as e:
#         logger.error(f"STT processing failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")


# === ROUTE 6: CHAT (THE BRAIN) (FREE) --- UPDATED --- ===
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, http_request: Request):
    if 'st_model' not in models:
        raise HTTPException(status_code=503, detail="Chat service (RAG models) is not ready.")

    try:
        # --- 1. RAG: Get Knowledge Base Context (Same as before) ---
        st_model = models['st_model']
        index = models['faiss_index']
        mapping = models['index_mapping']
        kb_data = models['kb_data_map']

        query_embedding = st_model.encode([request.query])
        k = 1 # Find the single best match
        distances, indices = index.search(query_embedding, k)

        best_match_text = "I'm sorry, I don't have specific information on that in my knowledge base."
        if indices.size > 0 and indices[0][0] != -1: # Check if index is valid
            faiss_index = indices[0][0]
            original_id = mapping.get(faiss_index) # Use the correct index
            if original_id:
                best_match_text = kb_data.get(original_id, best_match_text)
            else:
                 logger.warning(f"FAISS index {faiss_index} not found in mapping.")
        else:
             logger.info(f"No relevant KB entry found for query: {request.query}")


        # --- 2. Build Answer (Translate if needed, using mBART) ---
        answer_text = best_match_text # Start with the English KB text

        if request.language != "English":
            lang_details = LANG_CODES.get(request.language)
            # Check if language and mbart code exist
            if lang_details and lang_details.get("mbart"):
                mbart_lang_code = lang_details["mbart"]
                trans_model_name = "facebook/mbart-large-50-many-to-many-mmt"
                trans_model_url = f"{HF_API_BASE_URL}{trans_model_name}"
                trans_payload = {
                    "inputs": best_match_text,
                    "parameters": {
                        "src_lang": "en_XX",
                        "tgt_lang": mbart_lang_code
                    }
                }
                try:
                    logger.info(f"Translating chat answer to {request.language} ({mbart_lang_code}) using mBART...")
                    trans_response = hf_api_query(trans_payload, trans_model_url)
                    # Check response format
                    if isinstance(trans_response, list) and len(trans_response) > 0:
                        translated_answer = trans_response[0].get('translation_text')
                        if translated_answer:
                            answer_text = translated_answer # Update answer_text if translation succeeds
                        else:
                            logger.warning(f"mBART translation to {request.language} succeeded but response was empty. Using English.")
                    else:
                        logger.warning(f"Unexpected mBART translation response format: {trans_response}. Using English.")
                except Exception as e:
                    # Log the specific error from hf_api_query (which raises HTTPException)
                    logger.warning(f"mBART Translation to {request.language} failed: {e}. Using English answer.")
            else:
                 logger.warning(f"Language {request.language} not supported for mBART translation. Using English.")
        else:
             logger.info("Chat language is English, skipping translation.")


        # --- 3. Generate Audio for Answer ---
        # --- UPDATED --- Get gTTS code from the nested dictionary
        gtts_lang_code = LANG_CODES.get(request.language, {}).get("gtts", "en") # Default to 'en' if language not found
        relative_audio_path = generate_tts(answer_text, gtts_lang_code) # Pass gtts_lang_code
        audio_url = None
        if relative_audio_path:
            base_url = str(http_request.base_url).rstrip('/')
            audio_url = f"{base_url}{relative_audio_path}"


        # --- 4. Save Chat & Return ---
        try:
            with open("data/cases.json", "r+") as f:
                # Read existing history, handle potential empty file
                try:
                    history = json.load(f)
                    if not isinstance(history, list): history = [] # Ensure it's a list
                except json.JSONDecodeError:
                    history = [] # Start fresh if file is corrupt/empty

                history.append(
                    {"query": request.query, "answer": answer_text, "file_id": request.file_id, "audio_url": audio_url}
                )
                f.seek(0) # Go to start of file
                f.truncate() # Clear existing content before writing
                json.dump(history, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")

        return ChatResponse(
            answer_text=answer_text,
            audio_url=audio_url,
            file_id=request.file_id,
            query=request.query
        )

    except Exception as e:
        logger.error(f"Chat processing failed unexpectedly: {e}")
        # Add more specific error detail if possible
        detail = f"Chat processing failed: {str(e)}"
        if isinstance(e, HTTPException): # Propagate HTTPException details
             detail = e.detail
        raise HTTPException(status_code=500, detail=detail)

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    mimetypes.add_type("application/javascript", ".js")
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)