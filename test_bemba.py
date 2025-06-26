from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="NextInnoMind/next_bemba_ai")
result = pipe("sample_bemba.wav")  # Replace with your audio file
print(result["text"])
