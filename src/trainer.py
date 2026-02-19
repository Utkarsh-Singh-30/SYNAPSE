import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

# Import from your modules
from src.dataset import NIHDataset, LABELS
from src.model_factory import get_model

# ---------------- CONFIG ----------------
DATA_DIR = './data'
IMG_DIR = os.path.join(DATA_DIR, 'images')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'val.csv')
 
# Output paths
OUTPUT_BASE = './new_output'
MODEL_DIR = os.path.join(OUTPUT_BASE, 'models')
LOG_DIR = os.path.join(OUTPUT_BASE, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def get_pos_weights(dataset):
    # Calculate class weights to handle class imbalance
    df = dataset.df
    counts = df[LABELS].sum().values
    total = len(df)
    # Weight = (Negatives / Positives)
    weights = (total - counts) / (counts + 1e-5)
    return torch.tensor(weights, dtype=torch.float32)

def safe_auc(y_true, y_pred):
    try:
        # Micro-average AUC is standard for multi-label
        return roc_auc_score(y_true, y_pred, average='micro')
    except:
        return 0.5

def train_one_model(model_name, epochs=12, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*20} STARTING: {model_name} {'='*20}")

    # 1. PREPARE DATA LOADER
    # Strong augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Standard resize for validation
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = NIHDataset(TRAIN_CSV, IMG_DIR, transform_train)
    val_ds = NIHDataset(VAL_CSV, IMG_DIR, transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. SETUP MODEL
    model = get_model(model_name, len(LABELS), device)
    
    # Loss function with class weights
    pos_weights = get_pos_weights(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Logging
    log_file = os.path.join(LOG_DIR, f'{model_name}_log.csv')
    with open(log_file, 'w') as f:
        f.write('Epoch,Train_Loss,Val_Loss,Val_AUC,Time\n')

    best_auc = 0.0

    # 3. TRAINING LOOP
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch+1}/{epochs}')
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Sigmoid for AUC calculation
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate AUC
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        val_auc = safe_auc(all_labels, all_preds)
        
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1} | T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f} | AUC: {val_auc:.4f} | {elapsed:.0f}s")

        # Save Log
        with open(log_file, 'a') as f:
            f.write(f'{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_auc:.4f},{elapsed:.2f}\n')

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(MODEL_DIR, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved! (AUC: {best_auc:.4f})")