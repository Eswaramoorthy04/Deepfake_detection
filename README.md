# 🧠 Deepfake Detection System

A web application that detects whether an image or video is **REAL or FAKE** using a fine-tuned deep learning model.

---

## 🚀 Features

- 📷 Image deepfake detection
- 🎬 Video deepfake detection
- 🧑‍🦰 Face detection using MediaPipe
- 📊 Displays Fake %, Real %, and Confidence Score
- 🌐 Clean and user-friendly web interface

---

## 🧠 Model Details

- **Model:** SigLIP Vision Transformer
- **Framework:** PyTorch + HuggingFace Transformers
- **Fine-tuned on:** RVF10K Dataset (Kaggle)
- **Task:** Binary Classification — REAL vs FAKE

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI, Uvicorn |
| Face Detection | MediaPipe |
| Deep Learning | PyTorch |
| Model | SigLIP (fine-tuned) |

---

## 📂 Project Structure

```
Deepfake_detection/
├── main.py                   # FastAPI backend
├── index.html                # Frontend UI
├── app.js                    # Frontend logic
├── style.css                 # Styling
├── requirements.txt          # Dependencies
├── local_model/              # Fine-tuned model files
│   ├── model.safetensors
│   ├── config.json
│   └── preprocessor_config.json
└── Dataset/                  # RVF10K dataset
    ├── rvf10k/
    ├── train.csv
    └── valid.csv
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Eswaramoorthy04/Deepfake_detection.git
cd Deepfake_detection
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Start the backend**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**5. Open the frontend**

Double click `index.html` in your file explorer.

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict/image` | Upload image for detection |
| POST | `/predict/video` | Upload video for detection |

---

## 🧪 How It Works

**Image:**
1. Upload image → face detected by MediaPipe
2. Face cropped and resized to 224×224
3. SigLIP analyses facial features
4. Returns REAL / FAKE with confidence score

**Video:**
1. Upload video → frames extracted every 5th frame
2. MediaPipe detects face in each frame
3. SigLIP runs on each face crop
4. Scores averaged across all frames → final verdict

---

## 🎯 Accuracy

~85% to 92% after fine-tuning on RVF10K dataset

---

## 👨‍💻 Author

**Eswaramoorthy Sivathanu**
B.Tech — Artificial Intelligence and Data Science

---

> This project is developed for academic (final year) purposes.
