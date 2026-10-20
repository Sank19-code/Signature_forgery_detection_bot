# Signature Forgery Detection (EfficientNet-B2)

A clean, streamlined repository for detecting genuine vs. forged signatures using **Digital Image Processing (DIP)** and an **EfficientNet-B2** convolutional neural network. Includes a single Jupyter notebook for training/testing with CSV exports and a deployment-ready **Streamlit web application**.

---

## 📁 Repository Structure

```text
├── EfficientNet_B2_Signature_Classification.ipynb  # End-to-end training & testing notebook
├── app.py                                          # Streamlit web deployment application
├── pipeline.py                                     # Modular DIP preprocessing & model inference
├── requirements.txt                                # Project dependencies
├── models/
│   └── efficientnet_b2_signature.pt                # Saved EfficientNet-B2 model checkpoint
├── results/                                        # Generated metrics, CSV outputs & plots
│   ├── training_history.csv                        # Loss & validation history across epochs
│   ├── test_predictions.csv                       # Per-sample predictions on test images
│   ├── evaluation_metrics.csv                     # Accuracy, AUC, FAR, FRR summary metrics
│   ├── training_curves.png                         # Loss & accuracy/AUC curves
│   └── evaluation_plots.png                        # Confusion matrix and ROC curve
└── data/                                           # Dataset directory
    ├── train/ (genuine/, forged/)
    └── test/  (genuine/, forged/)
```

---

## ⚙️ Installation & Virtual Environment Setup

1. **Create and activate a virtual environment using Python's `venv` module:**
   ```bash
   # On macOS / Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Register the Jupyter kernel for the notebook:**
   ```bash
   python -m ipykernel install --user --name signature-env --display-name "Python (Signature Forgery .venv)"
   ```

---

## 🚀 Running the Notebook

Open and run [`EfficientNet_B2_Signature_Classification.ipynb`](file:///Users/abhijith/Documents/DIP/case%20study/Signature_forgery_detection_bot/EfficientNet_B2_Signature_Classification.ipynb):

1. **Automatic Dataset Discovery**:
   - Supports images located in `data/train` and `data/test` (subfolders `genuine` and `forged`).
   - Automatically unzips any `.zip` archive present in `data/` or the root folder.
   - If an unpartitioned dataset is provided, it automatically splits the signatures 80/20 into train/test sets with stratification.
2. **Four-Stage DIP Pipeline**:
   - Canonicalizes images (Otsu thresholding, ink level normalization, letterbox resizing).
   - Generates 3-channel feature representations using **Stage 2 (Median Filter)**, **Stage 3 (High-Boost Filtering)**, and **Stage 4 (Sobel Gradient Magnitude)**.
3. **Training & Checkpoint Saving**:
   - Trains `EfficientNet-B2` using AdamW and Cosine Annealing.
   - Evaluates after each epoch and automatically saves the best model checkpoint to [`models/efficientnet_b2_signature.pt`](file:///Users/abhijith/Documents/DIP/case%20study/Signature_forgery_detection_bot/models/efficientnet_b2_signature.pt).
4. **CSV Results & Evaluation Plots**:
   - Saves epoch-by-epoch loss & validation stats to `results/training_history.csv`.
   - Saves per-image test predictions (probabilities, predicted label, correct flag) to `results/test_predictions.csv`.
   - Saves summary metrics (Accuracy, AUC, FAR, FRR, Confusion Matrix statistics) to `results/evaluation_metrics.csv`.
   - Plots and exports training curves, confusion matrix, and ROC curve to `results/`.

---

## 🌐 Running the Streamlit App

Launch the interactive web application to upload and verify signatures in real-time:

```bash
streamlit run app.py
```

### Streamlit Features:
- **Instant Signature Verification**: Upload any signature image (PNG, JPG, BMP) to get a real-time classification (Genuine vs. Forged) with confidence score percentages.
- **Calibrated Threshold Support**: Uses the Equal Error Rate (EER) threshold saved in the model checkpoint or lets you interactively adjust the threshold via a slider.
- **Interactive DIP Stage Inspector**: Expandable view showing intermediate image processing steps (Canonical binarized image, Median filter, High-Boost filter, and Sobel gradient magnitude).
