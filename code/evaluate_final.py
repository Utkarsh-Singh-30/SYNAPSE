import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

# Custom Imports
from src.model_factory import get_model
from src.dataset import NIHDataset, LABELS

# ---------------- CONFIG ----------------
DATA_DIR = './data'
IMG_DIR = os.path.join(DATA_DIR, 'images')
VAL_CSV = os.path.join(DATA_DIR, 'val.csv')

OUTPUT_DIR = './new_output'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'final_reports')

os.makedirs(REPORT_DIR, exist_ok=True)

def generate_reports():
    print(f"\n{'='*20} GENERATING REPORTS (WITH RECALL & ACC) {'='*20}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Prepare Data Loader
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_ds = NIHDataset(VAL_CSV, IMG_DIR, transform_val)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)
    
    # 2. Find all saved best models
    model_files = glob.glob(os.path.join(MODEL_DIR, "*_best.pth"))
    
    if not model_files:
        print("[ERROR] No models found in 'new_output/models/'")
        return

    for model_path in model_files:
        filename = os.path.basename(model_path)
        model_name = filename.replace('_best.pth', '')
        
        print(f"--> Processing: {model_name}")
        
        try:
            model = get_model(model_name, len(LABELS), device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except Exception as e:
            print(f"    [SKIP] Load failed for {model_name}: {e}")
            continue

        # Run Inference
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"    Eval {model_name}", leave=False):
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
                
                y_pred.append(probs.cpu().numpy())
                y_true.append(labels.numpy())
        
        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # --- METRICS CALCULATION ---
        
        # 1. Detailed Report (Dictionary for extraction + String for file)
        report_dict = classification_report(y_true, y_pred_binary, target_names=LABELS, zero_division=0, output_dict=True)
        report_str = classification_report(y_true, y_pred_binary, target_names=LABELS, zero_division=0)
        
        # Extract Weighted Recall
        overall_recall = report_dict['weighted avg']['recall']

        # 2. Overall AUC
        try:
            auc = roc_auc_score(y_true, y_pred, average='micro')
        except:
            auc = 0.5
            
        # 3. Overall Accuracy (Element-wise)
        accuracy = (y_pred_binary == y_true).mean()

        # Save to Text File
        save_path = os.path.join(REPORT_DIR, f"{model_name}_detailed_report.txt")
        with open(save_path, "w") as f:
            f.write(f"MODEL ARCHITECTURE: {model_name}\n")
            f.write(f"OVERALL VALIDATION AUC: {auc:.4f}\n")
            f.write(f"OVERALL ACCURACY: {accuracy:.4f}\n")
            f.write(f"OVERALL RECALL: {overall_recall:.4f}\n") # <--- Added This
            f.write("="*60 + "\n")
            f.write("DETAILED CLASSIFICATION METRICS (Threshold = 0.5)\n")
            f.write("="*60 + "\n")
            f.write(report_str)
            f.write("\n" + "="*60 + "\n")
        
        print(f"    [DONE] Saved: {save_path} (Acc: {accuracy:.4f} | Rec: {overall_recall:.4f})")

if __name__ == "__main__":
    generate_reports()
    print("\n[SUCCESS] All reports generated.")