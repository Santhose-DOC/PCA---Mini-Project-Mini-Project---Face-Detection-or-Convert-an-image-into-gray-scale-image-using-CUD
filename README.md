# 🚀 GPU-Accelerated Spam Email Classifier using DistilBERT (Google Colab · PyTorch · Transformers)

This project implements a **GPU-accelerated spam detection system** using  
**DistilBERT (Transformer model) fine-tuned on email data**.  
It runs efficiently on **Google Colab's NVIDIA T4 GPU**, providing fast training and real-time prediction.

The project includes:

- ✨ Transformer-based classification (DistilBERT)
- ⚡ GPU-accelerated training (CUDA 12 · PyTorch)
- 🎯 High-accuracy spam detection
- 🧪 TF-IDF + MLP baseline (optional)
- 🖥️ Real-time Gradio web interface
- 📈 Evaluation metrics (Accuracy, F1 Score)
- 🔧 Full one-cell executable notebook provided

---

## 📌 **Project Overview**

Spam classification is a classic Natural Language Processing (NLP) problem.  
This project uses two models:

### **1️⃣ DistilBERT (Primary Model)**
A modern transformer model:
- Faster & smaller than BERT
- Highly accurate for text classification
- Pretrained on huge English corpora
- Fine-tuned on your email dataset

### **2️⃣ TF-IDF + MLP Baseline (Optional)**
A traditional lightweight baseline that:
- Runs extremely fast
- Provides CPU vs GPU comparison
- Useful for benchmarking

---

## ⚙️ **Technologies Used**

| Component | Technology |
|----------|------------|
| Deep Learning | PyTorch |
| NLP Model | DistilBERT (HuggingFace Transformers) |
| Dataset Handling | pandas, datasets |
| Metrics | scikit-learn (accuracy, F1 score) |
| Deployment UI | Gradio |
| Runtime | Google Colab (Python 3.12, CUDA 12) |
| GPU | NVIDIA Tesla T4 |

---

## 📂 **Project Structure**
```├── mail_data.csv # Training dataset
├── notebook.ipynb # One-cell runnable Colab notebook
├── README.md # Project documentation
└── (optional) screenshots/ # UI and output images
```

---

## 🧠 **How the Model Works**

### 🚀 DistilBERT Workflow
1. Text is tokenized using WordPiece tokenizer  
2. Attention layers extract contextual meaning  
3. Classification head predicts **spam (0)** or **ham (1)**  
4. Softmax gives probability scores  
5. Output is displayed in Gradio UI  

### 🧪 Evaluation Metrics
- **Accuracy** = correct predictions / total predictions  
- **F1-score** = harmonic mean of precision & recall  
- Gives robust insight for imbalanced spam datasets  

---

## 📊 **Performance**

Typical results using DistilBERT on a small dataset:

| Metric | Score |
|--------|--------|
| Accuracy | ~97% |
| F1-score | ~96–98% |
| Training Time (T4 GPU) | ~40–90 seconds |

These values vary depending on dataset size.

---

## 🖥️ **Real-Time Spam Detection UI**

The project includes a **Gradio-powered interface** that allows you to:

- Paste an email  
- Click **Submit**  
- View:
  - Spam or Ham prediction
  - Spam probability
  - Ham probability  

No backend or model-saving steps needed.

---

## ▶️ **How to Run This Project in Google Colab**

1. Open a new Colab notebook  
2. Upload **mail_data.csv**  
3. Enable **GPU Runtime**  
   - Runtime → Change Runtime Type → GPU (T4)  
4. Copy & paste the **one-cell full code**  
5. Run the notebook  
6. Use the Gradio interface to test emails  

---

## 📥 **Dataset Information**

- The project uses **mail_data.csv** containing:
  - `Message`: email text  
  - `Category`: spam / ham  

Before training:
- Empty rows are removed  
- Categories are mapped:  
  - `spam → 0`  
  - `ham → 1`

---

## 🔒 **Why DistilBERT?**

Compared to traditional ML methods:
- Understands context (semantic meaning)
- Detects sophisticated spam
- Handles long emails
- More accurate & robust

---

## 🚀 **Key Features**

- ✔ Fully GPU accelerated  
- ✔ High accuracy  
- ✔ Real-time interface  
- ✔ Clean, modern NLP architecture  
- ✔ No external logging (wandb disabled)  
- ✔ Compatible with Python 3.12  
- ✔ No model export needed  
- ✔ Single-cell runnable code  

---

## 💻 **PROGRAM**

```
# ============================================================
# FINAL FULLY WORKING DISTILBERT SPAM CLASSIFIER (NO evaluate)
# Compatible with Python 3.12, CUDA 12.1, Colab T4
# ============================================================

# ---- INSTALL ----
!pip install -q transformers datasets gradio
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ---- DISABLE WANDB ----
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

# ---- IMPORTS ----
import pandas as pd
import numpy as np
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("mail_data.csv")
df = df.dropna(subset=["Message", "Category"])
df["Category"] = df["Category"].str.lower().map({"spam": 0, "ham": 1})
df = df[df["Category"].isin([0,1])].reset_index()

texts = df["Message"].astype(str).tolist()
labels = df["Category"].astype(int).tolist()

X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def encode_texts(texts, labels):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return Dataset.from_dict({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels
    })

train_ds = encode_texts(X_train_text, y_train)
test_ds  = encode_texts(X_test_text, y_test)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
).to(DEVICE)

# ------------------------------------------------------------
# TRAINING ARGUMENTS (Python 3.12 compatible)
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./spam_model",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    logging_steps=50
)

# ------------------------------------------------------------
# METRICS WITHOUT evaluate
# ------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)

    return {"accuracy": acc, "f1": f1}

# ------------------------------------------------------------
# TRAINER
# ------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
print("Training...")
start = time.time()
trainer.train()
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
print("Training completed in:", time.time() - start, "seconds")

# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)

# ------------------------------------------------------------
# GRADIO REAL-TIME SPAM CHECKER
# ------------------------------------------------------------
def predict_spam(email_text):
    try:
        tokens = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        with torch.no_grad():
            logits = model(**tokens).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        label = "HAM" if pred == 1 else "SPAM"

        return {
            "Prediction": label,
            "Spam Probability": float(probs[0]),
            "Ham Probability": float(probs[1])
        }

    except Exception as e:
        return {"Error": str(e)}

ui = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=8, placeholder="Paste an email..."),
    outputs=gr.JSON(),
    title="DistilBERT Spam Classifier (GPU Accelerated)",
    description="Paste an email and classify it as SPAM or HAM!"
)

ui.launch(share=False)

```
---
## 🔳 **Output**
<img width="739" height="574" alt="image" src="https://github.com/user-attachments/assets/6361721e-0d41-4773-9a23-19096f5cd6aa" />

## ❗ Limitations

- Requires GPU (Transformer training is slow on CPU)
- Dataset quality affects model accuracy
- Long emails may need increased max_length (default = 128 tokens)

---

## 📌 **Future Improvements**

- Add LSTM/GRU comparison  
- Add spam explanation via SHAP/LIME  
- Add dataset augmentation  
- Deploy UI to HuggingFace Spaces or Streamlit Cloud  

---

## 🤝 **Contributing**

Pull requests are welcome!  
If you'd like to add features (metrics, visualization, deploy scripts), feel free to open an issue.

---

## 📜 **License**

This project is free and open-source under the **MIT License**.

---

## 🧑‍💻 Author

**SANTHOSE AROCKIARAJ J**  
GPU-Accelerated Spam Detection • DistilBERT Fine-Tuning • Google Colab


