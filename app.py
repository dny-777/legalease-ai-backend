import uvicorn
import os
import uuid
import json
import logging
import mimetypes
import requests
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Audio (Keep gTTS locally, it's light)
from gtts import gTTS

# Your API Schemas
from schemas import * # Assuming schemas.py is still present

# --- 1. INITIAL SETUP & CONFIG ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Hugging Face & ML Backend Config ---
HF_API_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
HF_API_BASE_URL = "https://api-inference.huggingface.co/models/"
ML_BACKEND_URL = os.getenv("ML_BACKEND_URL") # URL for your HF Space

# --- Create Static Dirs ---
os.makedirs("audio", exist_ok=True) # Only audio needed now

# --- 2. GLOBAL VARIABLES & HELPERS ---
# No local ML models needed here anymore
# models = {}

# Language codes (Keep this)
LANG_CODES = {
    "English":   {"gtts": "en", "mbart": "en_XX"},
    "Hindi":     {"gtts": "hi", "mbart": "hi_IN"},
    "Marathi":   {"gtts": "mr", "mbart": "mr_IN"},
    # ... keep other languages ...
}

def init_json_db(filepath: str, default_data):
     # Keep this helper if you still use local JSON for cases/uploads metadata
     os.makedirs(os.path.dirname(filepath), exist_ok=True)
     if not os.path.exists(filepath):
         logger.warning(f"Creating empty file: {filepath}")
         with open(filepath, "w") as f:
             json.dump(default_data, f, indent=4)

def save_upload_metadata(file_id, original_name, raw_text): # Simplified saver
    try:
        os.makedirs("data", exist_ok=True)
        filepath = "data/uploads.json"
        try:
            with open(filepath, "r+") as f:
                data = json.load(f)
                if not isinstance(data, dict): data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        data[file_id] = { "original_name": original_name, "raw_text": raw_text, "simplified": "", "translated": "" }
        with open(filepath, "w") as f: # Overwrite with updated data
             json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save upload metadata: {e}")


# Keep generate_tts
def generate_tts(text: str, gtts_lang_code: str) -> Optional[str]:
    if not gtts_lang_code: return None
    try:
        os.makedirs("audio", exist_ok=True)
        tts = gTTS(text=text, lang=gtts_lang_code, slow=False, tld='co.in')
        file_id = str(uuid.uuid4())
        relative_file_path = f"audio/{file_id}.mp3"
        tts.save(relative_file_path)
        return f"/{relative_file_path}"
    except Exception as e:
        logger.error(f"gTTS failed: {e}")
        return None

# Keep hf_api_query
def hf_api_query(payload, model_url):
    # ... (keep the existing hf_api_query function) ...
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
        if hasattr(errh, 'response') and errh.response is not None:
             status_code = errh.response.status_code
             detail_msg = errh.response.text
             try: detail_msg = errh.response.json()
             except json.JSONDecodeError: pass
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
    if not ML_BACKEND_URL:
        logger.error("ML_BACKEND_URL environment variable is not set!")
    # No heavy models to load here
    os.makedirs("data", exist_ok=True)
    init_json_db("data/uploads.json", {})
    init_json_db("data/cases.json", [])
    logger.info("Lightweight backend ready.")
    yield
    logger.info("Application shutdown...")


# --- 4. FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="LegalEase AI API (Lightweight Orchestrator)",
    description="Handles requests and calls ML backend for heavy tasks.",
    version="1.1.0", # Bump version
    lifespan=lifespan
)

# Mount only audio directory
try:
    app.mount("/audio", StaticFiles(directory="audio"), name="audio")
    # Don't mount /uploads if files aren't stored/served here long-term
except RuntimeError as mount_error:
     logger.error(f"Failed to mount static directory: {mount_error}")

app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 5. API ENDPOINTS (MODIFIED) ---

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the LegalEase AI Lightweight API!"}

# === ROUTE 1: OCR (Calls ML Backend) ===
@app.post("/api/upload_ocr", response_model=UploadOCRResponse, tags=["OCR"])
async def upload_and_ocr(file: UploadFile = File(...)):
    if not ML_BACKEND_URL:
        raise HTTPException(status_code=503, detail="ML Backend service URL not configured.")

    ocr_endpoint = f"{ML_BACKEND_URL.rstrip('/')}/ocr"
    file_id = str(uuid.uuid4())
    safe_filename = os.path.basename(file.filename or f"{file_id}_upload")

    try:
        files = {'file': (safe_filename, await file.read(), file.content_type)}
        logger.info(f"Sending file to ML backend for OCR: {ocr_endpoint}")
        response = requests.post(ocr_endpoint, files=files, timeout=120) # Increased timeout for OCR
        response.raise_for_status()
        ocr_data = response.json()
        raw_text = ocr_data.get("raw_text", "")

        # Save metadata locally (optional, could store in ML backend too)
        save_upload_metadata(file_id, safe_filename, raw_text)

        return UploadOCRResponse(file_id=file_id, file_name=safe_filename, raw_text=raw_text)

    except requests.exceptions.Timeout:
         logger.error(f"Timeout calling OCR endpoint: {ocr_endpoint}")
         raise HTTPException(status_code=504, detail="OCR request to ML backend timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling OCR endpoint {ocr_endpoint}: {e}")
        detail = f"Error communicating with ML backend: {e.response.text if e.response else str(e)}"
        status = e.response.status_code if e.response else 500
        raise HTTPException(status_code=status, detail=detail)
    except Exception as e:
        logger.error(f"Unexpected error during OCR orchestration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during OCR: {str(e)}")

# === ROUTE 2: SIMPLIFY (Calls HF API) ===
# (No change needed, already uses HF API)
@app.post("/api/simplify", response_model=SimplifyResponse, tags=["NLP"])
async def simplify_text(request: TextRequest):
    # ... keep existing simplify_text code using hf_api_query ...
    payload = {"inputs": f"summarize: {request.text}"}
    model_url = f"{HF_API_BASE_URL}sshleifer/distilbart-cnn-12-6"
    response_data = hf_api_query(payload, model_url)
    if isinstance(response_data, list) and len(response_data) > 0:
        simplified = response_data[0].get('summary_text')
        if simplified is not None:
             return SimplifyResponse(simplified_text=simplified)
    logger.error(f"Unexpected API response format for simplify: {response_data}")
    raise HTTPException(status_code=500, detail="API response format invalid or empty.")


# === ROUTE 3: TRANSLATE (Calls HF API - mBART) ===
# (No change needed, already uses HF API)
@app.post("/api/translate", response_model=TranslateResponse, tags=["NLP"])
async def translate_text(request: LanguageRequest):
    # ... keep existing translate_text code using hf_api_query and mBART ...
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


# === ROUTE 4: TEXT-TO-SPEECH (TTS) ===
# (No change needed, uses local gTTS)
@app.post("/api/tts", response_model=TTSResponse, tags=["Audio"])
async def text_to_speech(request: TTSRequest, http_request: Request):
    # ... keep existing text_to_speech code using generate_tts ...
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


# === ROUTE 5: SPEECH-TO-TEXT (STT) --- STILL DISABLED --- ===
# @app.post("/api/stt", ...)

# === ROUTE 6: CHAT (Calls ML Backend for RAG, HF API for Translate) ===
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, http_request: Request):
    if not ML_BACKEND_URL:
        raise HTTPException(status_code=503, detail="ML Backend service URL not configured.")

    search_endpoint = f"{ML_BACKEND_URL.rstrip('/')}/search"

    try:
        # --- 1. RAG: Call ML Backend to get best match ---
        logger.info(f"Sending query to ML backend for search: {search_endpoint}")
        # Send query text, ML backend handles embedding and searching
        search_payload = {"query": request.query, "k": 1}
        response = requests.post(search_endpoint, json=search_payload, timeout=30)
        response.raise_for_status()
        search_results = response.json()

        best_match_text = search_results.get("best_match_text")
        if not best_match_text:
             best_match_text = "I'm sorry, I don't have specific information on that in my knowledge base."
             logger.info(f"No relevant KB entry found by ML backend for query: {request.query}")

        # --- 2. Build Answer (Translate if needed, using HF mBART API) ---
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
                        if translated_answer: answer_text = translated_answer
                        else: logger.warning(f"mBART translation succeeded but empty. Using English.")
                    else: logger.warning(f"Unexpected mBART translation response. Using English.")
                except Exception as e: logger.warning(f"mBART Translation failed: {e}. Using English.")
            else: logger.warning(f"Language {request.language} not supported for mBART. Using English.")
        else: logger.info("Chat language is English, skipping translation.")

        # --- 3. Generate Audio for Answer (using local gTTS) ---
        gtts_lang_code = LANG_CODES.get(request.language, {}).get("gtts", "en")
        relative_audio_path = generate_tts(answer_text, gtts_lang_code)
        audio_url = None
        if relative_audio_path:
            base_url = str(http_request.base_url).rstrip('/')
            audio_url = f"{base_url}{relative_audio_path}"

        # --- 4. Save Chat & Return ---
        try:
            os.makedirs("data", exist_ok=True)
            filepath = "data/cases.json"
            try:
                with open(filepath, "r+") as f:
                    history = json.load(f)
                    if not isinstance(history, list): history = []
            except (FileNotFoundError, json.JSONDecodeError): history = []
            history.append({ "query": request.query, "answer": answer_text, "file_id": request.file_id, "audio_url": audio_url })
            with open(filepath, "w") as f: json.dump(history, f, indent=4)
        except Exception as e: logger.error(f"Failed to save chat history: {e}", exc_info=True)

        return ChatResponse( answer_text=answer_text, audio_url=audio_url, file_id=request.file_id, query=request.query )

    except requests.exceptions.Timeout:
         logger.error(f"Timeout calling RAG search endpoint: {search_endpoint}")
         raise HTTPException(status_code=504, detail="Chat search request to ML backend timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling RAG search endpoint {search_endpoint}: {e}")
        detail = f"Error communicating with ML backend: {e.response.text if e.response else str(e)}"
        status = e.response.status_code if e.response else 500
        raise HTTPException(status_code=status, detail=detail)
    except Exception as e:
        logger.error(f"Chat processing failed unexpectedly: {e}", exc_info=True)
        detail = f"Chat processing failed: {str(e)}"
        if isinstance(e, HTTPException): detail = e.detail
        raise HTTPException(status_code=500, detail=detail)


# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    # Use PORT env var for Render/Railway, default to 10000 locally
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)