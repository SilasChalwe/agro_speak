# Agro Speak

*Voice Assistant Digital Platform*

---

### 🚀 Features

* 🎧 Voice-to-Text (Bemba ASR fine-tuned Whisper model)
* 🧠 AI language understanding
* 🌱 Agriculture knowledgebase (farming tips, weather, pest info)
* 📱 Mobile-friendly design (planned)
* 🔊 Offline support for low-connectivity

---

### 📁 Repo Structure

```
Afro_Speak/
├── model/              # Fine-tuned models & checkpoints
├── data/               # Dataset & audio files
├── scripts/            # Training & inference scripts
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

---

### 🛠️ Setup

1. Clone repo:

   ```bash
   git clone git@github.com:SilasChalwe/agro_speak.git
   cd agro_speak
   ```

2. Create & activate environment:

   ```bash
   python3 -m venv bemba_env
   source bemba_env/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run training scripts in `scripts/`

---

### 🧪 Training

* Use `scripts/fine_tune_whisper.py` for Bemba ASR fine-tuning
* Checkpoints saved in `/checkpoints/`

---

### 🌍 Future Plans

* API integrations for farmer info & weather
* NLP chatbot support
* Mobile app UI
* Edge deployment (e.g. Raspberry Pi)

---

### 🤝 Contributing

Pull requests welcome! Open issues for major changes.

---

### 📜 License

MIT License

---

### 🙌 Author

**Silas Chalwe**
GitHub: [silaschalwe](https://github.com/SilasChalwe)
