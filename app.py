
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware  # Added missing import
from transformers import pipeline
import tempfile
import logging
import os
from datetime import datetime
import shutil
import json

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("bemba_api.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folder for uploaded audio exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploaded_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# JSON file to store training data
TRAINING_LOG = os.path.join(UPLOAD_DIR, "transcription_log.json")

# Initialize training log file if not exists
if not os.path.exists(TRAINING_LOG):
    with open(TRAINING_LOG, 'w') as f:
        json.dump([], f)

# Load model
logging.info("🚀 Starting Bemba transcription API...")
try:
    logging.info("Loading model 'NextInnoMind/next_bemba_ai'...")
    pipe = pipeline("automatic-speech-recognition", model="NextInnoMind/next_bemba_ai")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    raise

@app.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    client_host = request.client.host
    logging.info(f"🎧 Request from {client_host} | File type: {file.content_type}")

    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg"]:
        logging.warning(f"❌ Invalid file format: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid audio format. Use WAV or MP3.")

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
            logging.info(f"📁 Temp saved to {tmp_path}")

        # Save permanent copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        final_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(tmp_path, final_path)
        logging.info(f"💾 Audio saved as: {final_path}")

        # Transcribe
        logging.info("🤖 Transcribing...")
        result = pipe(tmp_path)
        transcription = result["text"]
        logging.info(f"✅ Transcription done: {transcription[:60]}...")

        # Clean up temp file
        os.unlink(tmp_path)
        logging.info(f"🗑️ Deleted temp file: {tmp_path}")

        # Save metadata for future model fine-tuning
        record = {
            "timestamp": timestamp,
            "client_ip": client_host,
            "filename": filename,
            "transcription": transcription
        }

        _save_training_data(record)

        return {"transcription": transcription}

    except Exception as e:
        logging.error(f"❌ Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _save_training_data(record: dict):
    """Append the transcription + audio filename to the training log JSON file."""
    try:
        # Load existing data
        with open(TRAINING_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        data.append(record)

        # Save updated list
        with open(TRAINING_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logging.info(f"📄 Transcription logged for training: {record['filename']}")

    except Exception as e:
        logging.error(f"⚠️ Failed to save transcription log: {e}")
