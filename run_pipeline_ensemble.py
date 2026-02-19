import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageStat, ImageFilter

# Custom Modules
from src.model_factory import get_model
from src.grad_cam import generate_gradcam
from src.findings_mapper import VisualFindingMapper
from src.kg_reasoner import KGReasoner
from src.explanation_generator import ExplanationGenerator
from src.dataset import LABELS

# ---------------- CONFIGURATION ----------------
# Final Ensemble Team
MODEL_CONFIGS = [
    {'name': 'densenet121',       'path': './new_output/models/densenet121_best.pth'},       # Primary
    {'name': 'seresnext50_32x4d', 'path': './new_output/models/seresnext50_32x4d_best.pth'}, # High AUC
    {'name': 'resnet101',         'path': './new_output/models/resnet101_best.pth'}          # Balanced
]

# Threshold (0.35 is balanced for Recall/Safety)
FINAL_THRESHOLD = 0.35 

CSV_PATH = "./data/Data_Entry_2017.csv" 
TEST_IMAGE = "./data/images/00000001_000.png"
# TEST_IMAGE = "./data/images/00000005_000.png"
# TEST_IMAGE = "./data/images/00000022_000.png"
# TEST_IMAGE = "./data/normal/tree.png"
# TEST_IMAGE = "./data/normal/IM-0141-0001.png"
# TEST_IMAGE = "./data/normal/NORMAL2.png"
OUTPUT_DIR = "./new_output/reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- 1. GATEKEEPER (The Tree Killer) ----------------
def is_valid_xray(img_path):
    if not os.path.exists(img_path): return False, f"File not found: {img_path}"
    try:
        img = Image.open(img_path).convert('RGB')
        
        # 1. Color Check
        hsv = img.convert('HSV')
        if np.array(hsv)[:, :, 1].mean() > 25:
            return False, "Image is too colorful. Real X-rays are grayscale."

        # 2. Bottom Brightness Check (Photos usually have bright ground)
        gray_img = img.convert('L')
        img_arr = np.array(gray_img)
        if np.mean(img_arr[-20:, :]) > 240: 
             return False, "Bottom edge is unusually bright (Looks like a photo/sand)."

        # 3. Edge Density Check (Reject High Texture/Trees)
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edge_score = np.mean(np.array(edges))
        
        if edge_score > 25: 
            return False, f"Image is too 'noisy' or textured (Score: {edge_score:.1f}). X-rays are smoother."

        # 4. Blur/Blank Check
        variance = ImageStat.Stat(gray_img).var[0]
        if variance < 100:
            return False, "Image is too flat or blurry."

        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ---------------- 2. MAIN PIPELINE ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 🏥 NEURO-SYMBOLIC ENSEMBLE SYSTEM ({device}) ---")

    # [Step 0] Validate Image
    print(f"[0] Validating Image: {TEST_IMAGE}")
    valid, msg = is_valid_xray(TEST_IMAGE)
    if not valid:
        print("\n" + "="*50)
        print(f"🛑 REJECTED: {msg}")
        print("="*50)
        return

    # [Step 1] Load Models
    print(f"⚙️  Loading Ensemble Models...")
    models = []
    try:
        for config in MODEL_CONFIGS:
            if not os.path.exists(config['path']):
                print(f"   ⚠️  Missing Model: {config['path']} (Skipping)")
                continue
            
            m = get_model(config['name'], len(LABELS), device)
            m.load_state_dict(torch.load(config['path'], map_location=device))
            m.eval()
            models.append(m)
            
        if not models:
            print("❌ CRITICAL ERROR: No models loaded!")
            return
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # [Step 2] Ensemble Inference
    print(f"🧠 Running Analysis on {len(models)} Architectures...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    avg_probs = np.zeros(len(LABELS))
    with torch.no_grad():
        for m in models:
            avg_probs += torch.sigmoid(m(img_tensor))[0].cpu().numpy()
    
    avg_probs /= len(models) # Average
    model_probs = {LABELS[i]: float(avg_probs[i]) for i in range(len(LABELS))}

    # [Step 3] Visual Explainability
    print("👁️  Generating Grad-CAM...")
    IMG_OUT = os.path.join(OUTPUT_DIR, "images")
    heatmap = generate_gradcam(models[0], img_tensor, TEST_IMAGE, IMG_OUT)
    
    mapper = VisualFindingMapper()
    findings, location = mapper.map_findings(heatmap)

    # [Step 4] Neuro-Symbolic Logic Reasoner
    print("🤖 Applying Knowledge Graph Rules...")
    kg = KGReasoner()
    df = pd.read_csv(CSV_PATH) # CSV load karo
    
    # Image ka naam nikalo (path se)
    img_name = os.path.basename(TEST_IMAGE) 
    
    # Check karo agar image CSV mein hai
    if img_name in df['Image Index'].values:
        row = df[df['Image Index'] == img_name].iloc[0] # Woh row nikalo
        
        patient_data = {
            "id": str(row['Patient ID']),
            "age": int(row['Patient Age']),
            "gender": row['Patient Gender'] # 'M' or 'F'
        }
        print(f"   [INFO] Patient Found: Age={patient_data['age']}, Gender={patient_data['gender']}")
    else:
        # Agar image CSV mein nahi hai (External image), to default use karo
        print("   [WARN] Patient not found in CSV. Using defaults.")
        patient_data = {"id": "Unknown", "age": 0, "gender": "Unknown"} 


    # Ab ye asli data 'reason' function mein jayega
    final_diag, final_conf, trace, status_code = kg.reason(findings, model_probs, patient_data)

    # [Step 5] Final Report Generation
    print("📝 Generating Medical Report...")
    explainer = ExplanationGenerator()
    report = explainer.generate(findings, final_diag, final_conf, trace, status_code, location)

    # --- FIX: OVERRIDE REPORT TEXT FOR HEALTHY CASES ---
    # Agar status_code 0 (Normal) hai, to "Doctor Review" hata kar "Healthy" likho
    # if status_code == 0:
    #     report = report.replace("Recommendation: DOCTOR REVIEW REQUIRED.", "Recommendation: NO ACTION NEEDED (Routine Checkup).")
    #     report = report.replace("Recommendation: URGENT DOCTOR CONSULTATION REQUIRED.", "Recommendation: NO ACTION NEEDED (Routine Checkup).")
        
    #     # Kabhi kabhi text "Abnormality Detected" reh jata hai, use bhi clean karo
    #     if "UNSPECIFIED ABNORMALITY" in report:
    #          report = report.replace("UNSPECIFIED ABNORMALITY DETECTED", "NO ABNORMALITIES DETECTED")
    # # ---------------------------------------------------

    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    with open(f"{OUTPUT_DIR}/ensemble_final_report.txt", "w") as f:
        f.write(report)
    print(f"\n✅ Report Saved: {OUTPUT_DIR}/ensemble_final_report.txt")

if __name__ == "__main__":
    main()