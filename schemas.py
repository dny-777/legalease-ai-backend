from pydantic import BaseModel, Field
from typing import List, Optional

# === Base Models ===
class TextRequest(BaseModel):
    text: str = Field(..., example="This is a sample text.")

class LanguageRequest(TextRequest):
    target_language: str = Field(..., example="Hindi")

# === OCR ===
class UploadOCRResponse(BaseModel):
    file_id: str
    file_name: str
    raw_text: str

# === NLP ===
class SimplifyResponse(BaseModel):
    simplified_text: str

class TranslateResponse(BaseModel):
    translated_text: str

# === Audio ===
class TTSRequest(BaseModel):
    text: str
    language: str = Field(..., example="Hindi") # Must be a language name

class TTSResponse(BaseModel):
    audio_url: str

class STTResponse(BaseModel):
    transcribed_text: str

# === Chat (The Main Event) ===
class ChatRequest(BaseModel):
    query: str
    file_id: Optional[str] = None
    language: str = Field(default="English", example="English")

class ChatResponse(BaseModel):
    answer_text: str
    audio_url: Optional[str] = None
    file_id: Optional[str] = None
    query: str