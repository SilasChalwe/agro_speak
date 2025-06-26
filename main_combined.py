from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, VitsModel, AutoTokenizer
import torch
import io
import os
import subprocess
import tempfile
import scipy.io.wavfile as wavfile
import base64
import logging
from rich.logging import RichHandler
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler()]
)
logger = logging.getLogger("combined_api")

# Initialize FastAPI
app = FastAPI(title="Speech-to-Speech with Transcription")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend domain here for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ASR and TTS models
logger.info("Loading Whisper (ASR) and VITS (TTS) models...")
asr_pipeline = pipeline("automatic-speech-recognition", model="NextInnoMind/next_bemba_ai", chunk_length_s=30)
tts_model = VitsModel.from_pretrained("facebook/mms-tts-bem")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-bem")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tts_model = tts_model.to(device)
logger.info(f"Models loaded on {device}")

def convert_to_wav(input_path: str, output_path: str):
    """Convert any audio file to 16kHz mono WAV using ffmpeg."""
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-acodec", "pcm_s16le", "-f", "wav", output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

@app.post("/transcribe-and-speak")
async def transcribe_and_speak(file: UploadFile = File(...)):
    # Check file type
    if file.content_type not in ["audio/wav", "audio/mp3", "audio/m4a", "audio/x-wav"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    tmp_in = None
    tmp_out = None
    try:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_in_file:
            tmp_in_file.write(await file.read())
            tmp_in_file.flush()
            tmp_in = tmp_in_file.name

        # Convert to WAV 16kHz mono
        tmp_out = tmp_in + "_converted.wav"
        convert_to_wav(tmp_in, tmp_out)

        # Transcribe audio -> text
        transcription_result = asr_pipeline(tmp_out)
        transcription = transcription_result["text"].strip()
        logger.info(f"Transcription: {transcription}")

        # Generate speech from text
        inputs = tts_tokenizer(transcription, return_tensors="pt").to(device)
        with torch.no_grad():
            output = tts_model(**inputs)

        waveform = output.waveform.squeeze().cpu().numpy()

        # Convert waveform to WAV bytes in-memory
        buffer = io.BytesIO()
        wavfile.write(buffer, rate=tts_model.config.sampling_rate, data=waveform)
        buffer.seek(0)
        audio_bytes = buffer.read()

        # Encode audio bytes to base64 string
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Return JSON with text + audio base64
        return JSONResponse({
            "text": transcription,
            "audio_base64": audio_b64,
            "audio_format": "wav",
            "sample_rate": tts_model.config.sampling_rate
        })

    except subprocess.CalledProcessError as e:
        logger.error(f"Audio conversion error: {e}")
        raise HTTPException(status_code=500, detail="Audio conversion failed")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")
    finally:
        # Clean up temp files
        for path in [tmp_in, tmp_out]:
            if path and os.path.exists(path):
                os.unlink(path)

if __name__ == "__main__":
    uvicorn.run("main_combined:app", host="0.0.0.0", port=8000)
