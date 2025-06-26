# AgroSpeak.py

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline, VitsModel, AutoTokenizer
import torch
import io
import base64
import httpx
import scipy.io.wavfile as wavfile
import tempfile
from rich.logging import RichHandler
import uvicorn
from datetime import datetime

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler()]
)
logger = logging.getLogger("AgroSpeak")

app = FastAPI(title="AgroSpeak: Bemba STT + OpenRouter Chat + TTS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load speech-to-text pipeline (replace model with your actual Bemba ASR model)
logger.info("Loading speech-to-text model...")
stt_pipe = pipeline("automatic-speech-recognition", model="NextInnoMind/next_bemba_ai", chunk_length_s=30)
logger.info("STT model loaded.")

# Load TTS model and tokenizer
logger.info("Loading TTS model...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-bem")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-bem")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts_model = tts_model.to(device)
logger.info(f"TTS model loaded on {device}.")

# OpenRouter API settings - replace with your key
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

async def openrouter_chat_completion(prompt: str) -> str:
    # Request GPT reply in Bemba explicitly
    bemba_prompt = (
        f"Please reply in the Bemba language only. "
        f"Here is the user's message: {prompt}"
    )
    logger.info(f"Sending prompt to OpenRouter (in Bemba): {bemba_prompt[:50]}{'...' if len(bemba_prompt) > 50 else ''}")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",  # adjust as needed
        "messages": [
            {"role": "user", "content": bemba_prompt}
        ]
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        logger.info(f"Received response from OpenRouter: {text[:50]}{'...' if len(text) > 50 else ''}")
        return text

def synthesize_tts(text: str) -> bytes:
    logger.info(f"Synthesizing TTS for text: {text[:50]}{'...' if len(text) > 50 else ''}")
    inputs = tts_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = tts_model(**inputs)
    waveform = output.waveform.squeeze().cpu().numpy()

    buffer = io.BytesIO()
    wavfile.write(buffer, rate=tts_model.config.sampling_rate, data=waveform)
    logger.info("TTS audio generated.")
    return buffer.getvalue()

@app.post("/transcribe-chat-tts")
async def transcribe_chat_tts(file: UploadFile = File(...)):
    logger.info(f"Received audio upload: {file.filename}, Content-Type: {file.content_type}")
    if file.content_type != "audio/wav":
        logger.error("Invalid audio format, only WAV accepted.")
        raise HTTPException(status_code=400, detail="Only WAV audio format accepted")

    audio_bytes = await file.read()

    # Save audio temporarily to pass to STT pipeline
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            stt_result = stt_pipe(tmp.name)
            transcription = stt_result.get("text", "").strip()
    except Exception as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"STT transcription failed: {str(e)}")

    if not transcription:
        logger.error("Empty transcription received from STT model.")
        raise HTTPException(status_code=400, detail="Could not transcribe audio")

    logger.info(f"Transcription result: {transcription}")

    # Call OpenRouter chat
    try:
        chat_response = await openrouter_chat_completion(transcription)
    except Exception as e:
        logger.error(f"OpenRouter API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenRouter API error: {str(e)}")

    # Generate TTS audio for chat response
    try:
        audio_data = synthesize_tts(chat_response)
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

    audio_b64 = base64.b64encode(audio_data).decode("utf-8")

    logger.info(f"Request handled successfully at {datetime.now().isoformat()}")

    return JSONResponse({
        "transcription": transcription,
        "chat_response": chat_response,
        "audio_base64": audio_b64,
    })

if __name__ == "__main__":
    logger.info("Starting AgroSpeak API server on 0.0.0.0:8000")
    uvicorn.run("AgroSpeak:app", host="0.0.0.0", port=8000, reload=False)
