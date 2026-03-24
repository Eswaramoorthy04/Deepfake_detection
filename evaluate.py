import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------- CONFIG ----------------
MODEL_PATH = r"D:\Deepfake Detection\local_model"
DATASET_PATH = r"D:\Deepfake Detection\Dataset\rvf10k\valid"

# ---------------- LOAD MODEL ----------------
print("Loading model...")
model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

model.eval()

# ---------------- STORAGE ----------------
y_true = []
y_pred = []

# ---------------- PROCESS IMAGES ----------------
print("Evaluating...\n")

for label_name in ["real", "fake"]:
    folder = os.path.join(DATASET_PATH, label_name)
    images = os.listdir(folder)

    for img_name in tqdm(images, desc=f"Processing {label_name}"):
        img_path = os.path.join(folder, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            pred = torch.argmax(outputs.logits, dim=1).item()

            y_pred.append(pred)

            # Assign labels
            if label_name == "real":
                y_true.append(1)
            else:
                y_true.append(0)

        except Exception as e:
            print(f"Error with {img_name}: {e}")
            continue

# ---------------- METRICS ----------------
print("\n✅ Evaluation Complete!\n")

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()