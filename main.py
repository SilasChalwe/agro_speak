from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from rich.logging import RichHandler
import tempfile
import logging
import os
from datetime import datetime
import shutil
import json
import subprocess
from pathlib import Path

# Setup color logging using rich
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[RichHandler()]
)

logger = logging.getLogger("bemba_api")
app = FastAPI(
    title="Bemba Transcription API",
    max_upload_size=100 * 1024 * 1024  # 100MB upload limit
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)

TRAINING_LOG = os.path.join(UPLOAD_DIR, "transcription_log.json")
if not os.path.exists(TRAINING_LOG):
    with open(TRAINING_LOG, 'w') as f:
        json.dump([], f)

# Load Whisper pipeline
logger.info("🚀 Starting Bemba transcription API...")
try:
    logger.info("Loading model 'NextInnoMind/next_bemba_ai'...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model="NextInnoMind/next_bemba_ai",
        chunk_length_s=30
    )
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        logger.error(f"❌ Failed to get duration: {result.stdout}")
        return 0.0

def convert_to_wav(input_path: str, output_path: str):
    """Force audio to 16kHz mono 16-bit PCM WAV using high-quality resampling"""
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",          # Output sample rate
        "-ac", "1",              # Mono channel
        "-acodec", "pcm_s16le",  # 16-bit depth
        "-af", "aresample=resampler=soxr",  # High-quality resampler
        "-f", "wav",
        output_path
    ]
    try:
        subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FFmpeg conversion failed: {e.stderr}")
        raise RuntimeError("Audio conversion failed")

@app.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    client_host = request.client.host
    logger.info(f"🎧 Request from {client_host} | File: {file.filename} | Type: {file.content_type}")

    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/m4a"]:
        logger.warning(f"❌ Unsupported audio format: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid audio format. Use WAV, MP3, or M4A.")

    tmp_path = None
    converted_path = None
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            logger.info(f"📁 Temp file saved: {tmp_path}")

        # Convert to a proper WAV format
        converted_path = f"{tmp_path}_converted.wav"
        convert_to_wav(tmp_path, converted_path)
        
        # Validate audio duration
        duration = get_audio_duration(converted_path)
        if duration > 1800:  # 30 minutes
            logger.warning(f"❌ Audio too long: {duration}s")
            raise HTTPException(status_code=400, detail="Audio too long (max 30 minutes)")
        
        # Save permanent copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        final_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(converted_path, final_path)
        logger.info(f"💾 Audio saved as: {final_path}")

        # Transcribe
        logger.info("🤖 Transcribing...")
        result = pipe(converted_path)
        transcription = result["text"].strip()
        logger.info(f"✅ Transcription: {transcription[:60]}{'...' if len(transcription) > 60 else ''}")

        # Save to training log
        record = {
            "timestamp": timestamp,
            "client_ip": client_host,
            "filename": filename,
            "transcription": transcription,
            "corrected": False,
            "duration": duration
        }
        _save_training_data(record)

        return {
            "text": transcription,
            "filename": filename,
            "duration": f"{duration:.2f}s"
        }

    except (subprocess.CalledProcessError, RuntimeError) as e:
        logger.error(f"❌ Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Audio processing failed")
    except Exception as e:
        logger.exception(f"🔥 Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up temp files
        for path in [tmp_path, converted_path]:
            if path and os.path.exists(path):
                os.unlink(path)
                logger.debug(f"🗑️ Removed temp file: {path}")

def _save_training_data(record: dict):
    try:
        with open(TRAINING_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.append(record)

        with open(TRAINING_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 Transcription log updated for {record['filename']}")
    except Exception as e:
        logger.error(f"⚠️ Failed to update transcription log: {e}")

@app.post("/correct")
async def correct_transcription(
    filename: str = Body(...),
    corrected_text: str = Body(...)
):
    """User submits corrected transcription for a given filename"""
    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    
    try:
        if not os.path.exists(TRAINING_LOG):
            raise HTTPException(status_code=404, detail="No training log found.")

        with open(TRAINING_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)

        updated = False
        for item in data:
            if item["filename"] == safe_filename:
                item["transcription"] = corrected_text
                item["corrected"] = True
                updated = True
                break

        if not updated:
            raise HTTPException(status_code=404, detail="Filename not found in logs.")

        with open(TRAINING_LOG, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"✏️ Correction updated for {safe_filename}")
        return {"message": "Correction saved successfully."}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to save correction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save correction.")

# Run with Uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
