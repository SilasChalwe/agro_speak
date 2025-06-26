from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VitsModel, AutoTokenizer
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import io
import scipy.io.wavfile as wavfile
import logging
from rich.logging import RichHandler
from datetime import datetime
import uvicorn

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("uvicorn")

# Load the Bemba TTS model
logger.info("🔊 Loading Bemba TTS model...")
model = VitsModel.from_pretrained("facebook/mms-tts-bem")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-bem")

# Move model to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"✅ Model loaded on: {device}")

# Define FastAPI app
app = FastAPI(title="Bemba Text-to-Speech API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with allowed domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class TTSRequest(BaseModel):
    text: str

# POST endpoint
@app.post("/api/tts/bemba")
async def synthesize_bemba_tts(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info(f"[{datetime.now().isoformat()}] 🎤 Received text: {request.text}")

    # Tokenize and run model
    inputs = tokenizer(request.text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)

    waveform = output.waveform.squeeze().cpu().numpy()

    # Convert to WAV
    buffer = io.BytesIO()
    sample_rate = model.config.sampling_rate
    wavfile.write(buffer, rate=sample_rate, data=waveform)
    buffer.seek(0)

    logger.info("✅ Audio generated and ready to stream")
    return StreamingResponse(buffer, media_type="audio/wav")

# Auto-run the server if script is executed directly
if __name__ == "__main__":
    uvicorn.run("main_2:app", host="0.0.0.0", port=8000, reload=False)
