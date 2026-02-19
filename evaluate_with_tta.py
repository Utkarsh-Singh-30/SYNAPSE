import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Custom Imports
from src.model_factory import get_model
from src.dataset import NIHDataset, LABELS

# ---------------- CONFIG ----------------
DATA_DIR = './data'
IMG_DIR = os.path.join(DATA_DIR, 'images')
VAL_CSV = os.path.join(DATA_DIR, 'val.csv')
MODEL_PATH = './new_output/models/densenet121_best.pth' 
REPORT_DIR = './new_output/final_reports'
os.makedirs(REPORT_DIR, exist_ok=True)

def tta_inference(model, images):
    """
    Runs inference 3 times (Removed Flip to save Cardiomegaly score):
    1. Original
    2. Rotate +5 degrees
    3. Rotate -5 degrees
    Returns the average probabilities.
    """
    # 1. Original
    output1 = torch.sigmoid(model(images))
    
    # REMOVED HORIZONTAL FLIP (Bad for Heart/Lung side detection)
    
    # 2. Rotate +5
    images_rot5 = TF.rotate(images, 5)
    output2 = torch.sigmoid(model(images_rot5))
    
    # 3. Rotate -5
    images_rot_minus5 = TF.rotate(images, -5)
    output3 = torch.sigmoid(model(images_rot_minus5))
    
    # Average predictions (Divide by 3 now)
    avg_output = (output1 + output2 + output3) / 3.0
    return avg_output

def evaluate_tta():
    print(f"\n{'='*20} RUNNING SMART TTA (NO FLIP) {'='*20}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Data Loader
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_ds = NIHDataset(VAL_CSV, IMG_DIR, transform_val)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}")
        return

    try:
        model = get_model('densenet121', len(LABELS), device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 3. Inference Loop
    y_true = []
    y_pred = []
    
    print("Running TTA Inference (3x Pass)...")
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            imgs = imgs.to(device)
            probs = tta_inference(model, imgs)
            y_pred.append(probs.cpu().numpy())
            y_true.append(labels.numpy())
    
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 4. Calculate Metrics
    print("Calculating Metrics...")
    
    # Accuracy
    accuracy = (y_pred_binary == y_true).mean()
    
    # Classification Report
    report_dict = classification_report(y_true, y_pred_binary, target_names=LABELS, zero_division=0, output_dict=True)
    report_str = classification_report(y_true, y_pred_binary, target_names=LABELS, zero_division=0)
    
    # Extract Overall Recall (Weighted Avg)
    overall_recall = report_dict['weighted avg']['recall']

    # AUC
    try:
        auc = roc_auc_score(y_true, y_pred, average='micro')
    except:
        auc = 0.5

    # 5. Save Report
    save_path = os.path.join(REPORT_DIR, "densenet121_TTA_detailed_report.txt")
    
    with open(save_path, "w") as f:
        f.write(f"MODEL ARCHITECTURE: DenseNet121 + TTA\n") 
        f.write(f"OVERALL VALIDATION AUC: {auc:.4f}\n")
        f.write(f"OVERALL ACCURACY: {accuracy:.4f}\n")
        f.write(f"OVERALL RECALL: {overall_recall:.4f}\n") # <--- Added this line
        f.write("="*60 + "\n")
        f.write("DETAILED CLASSIFICATION METRICS (Threshold = 0.5)\n")
        f.write("="*60 + "\n")
        f.write(report_str)
        f.write("\n" + "="*60 + "\n")
    
    print(f"\n[SUCCESS] Report Saved: {save_path}")
    print(f"--> AUC: {auc:.4f}")
    print(f"--> Accuracy: {accuracy:.4f}")
    print(f"--> Recall: {overall_recall:.4f}")

if __name__ == "__main__":
    evaluate_tta()