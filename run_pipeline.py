# import os
# import torch
# import pandas as pd
# import numpy as np
# from torchvision import transforms
# from PIL import Image

# # Import custom modules
# from src.model_factory import get_model
# from src.grad_cam import generate_gradcam
# from src.findings_mapper import VisualFindingMapper
# from src.kg_reasoner import KGReasoner
# from src.explanation_generator import ExplanationGenerator
# from src.dataset import LABELS


# # ---------------- CONFIG ----------------
# # Use the best model you trained (check outputs/models/)
# # MODEL_NAME = 'thoraxnet' 
# # MODEL_PATH = f"./new_output/models/{MODEL_NAME}_best.pth"

# # Test Image (Pick one from your data folder)
# # TEST_IMAGE = "./data/images/00000001_000.png" 
# # OUTPUT_DIR = "./new_output/reports"

# # os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------------- CONFIGURATION ----------------
# MODEL_ARCH = 'densenet169'
# MODEL_PATH = "./models/densenet169_epoch_10.pth"
# CSV_PATH = "./data/Data_Entry_2017.csv" 

# # CHANGE THIS TO TEST DIFFERENT IMAGES
# TEST_IMAGE = "./data/images/00000001_000.png"
# # TEST_IMAGE = "./random_google_image.jpg" # Try this later

# OUTPUT_DIR = "./new_output/reports"
# os.makedirs(OUTPUT_DIR, exist_ok=True)




# # ---------------- METADATA LOOKUP ----------------
# def get_patient_metadata(image_path, csv_path):
#     """
#     Fetches patient details from the CSV based on filename.
#     Returns default values if image is not in CSV (e.g., downloaded from Google).
#     """
#     filename = os.path.basename(image_path)
    
#     # Default (Unknown Patient)
#     metadata = {
#         "id": "UNKNOWN",
#         "age": 0,
#         "gender": "Unknown",
#         "symptoms": [] # NIH dataset has no symptoms, so this is always empty/manual
#     }

#     if not os.path.exists(csv_path):
#         print(f"[WARNING] CSV not found at {csv_path}. Using default metadata.")
#         return metadata

#     try:
#         df = pd.read_csv(csv_path)
#         # Filter by filename
#         row = df[df['Image Index'] == filename]
        
#         if not row.empty:
#             # Found in Database
#             metadata["id"] = str(row.iloc[0]['Patient ID'])
#             metadata["age"] = int(row.iloc[0]['Patient Age'])
#             metadata["gender"] = row.iloc[0]['Patient Gender']
#             print(f"   -> Found Patient Record: ID {metadata['id']}, {metadata['age']}y, {metadata['gender']}")
#         else:
#             # Not in Database (Random Image)
#             print(f"   -> Image '{filename}' not found in CSV. Treated as new/external patient.")
            
#     except Exception as e:
#         print(f"[ERROR] Reading CSV: {e}")

#     return metadata

# # ---------------- MAIN PIPELINE ----------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"--- RUNNING SYNAPSE PIPELINE ({device}) ---")

#     # 1. FETCH METADATA (Replaces Hardcoded Data)
#     print(f"[1] Fetching Patient Data for: {TEST_IMAGE}")
#     patient_data = get_patient_metadata(TEST_IMAGE, CSV_PATH)
    
#     # OPTIONAL: You can manually simulate symptoms here since CSV doesn't have them
#     # patient_data['symptoms'] = ["Fever", "Cough"] 

#     # 2. LOAD MODEL
#     print(f"[2] Loading Model: {MODEL_ARCH}")
#     try:
#         model = get_model(MODEL_ARCH, len(LABELS), device)
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#         model.eval()
#     except Exception as e:
#         print(f"[ERROR] Failed to load model: {e}")
#         return

#     # 3. PROCESS IMAGE
#     print(f"[3] Processing Image...")
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     try:
#         raw_img = Image.open(TEST_IMAGE).convert("RGB")
#         img_tensor = transform(raw_img).unsqueeze(0).to(device)
#     except Exception as e:
#         print(f"[ERROR] Image load failed: {e}")
#         return

#     # 4. INFERENCE
#     with torch.no_grad():
#         logits = model(img_tensor)
#         probs = torch.sigmoid(logits)[0].cpu().numpy()
    
#     model_probs = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    
#     # # 5. VISUAL ATTENTION
#     # print("[4] Generating Heatmaps...")
#     # try:
#     #     heatmap = generate_gradcam(model, img_tensor, TEST_IMAGE, OUTPUT_DIR)
#     # except:
#     #     heatmap = np.zeros((224, 224)) # Fallback

# # 5. VISUAL ATTENTION
#     print("[4] Generating Heatmaps...")
    
#     # Explicitly define the image folder
#     IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
#     if not os.path.exists(IMG_OUTPUT_DIR):
#         os.makedirs(IMG_OUTPUT_DIR)
#         print(f"    [INFO] Created folder: {IMG_OUTPUT_DIR}")

#     try:
#         # Pass the SPECIFIC image folder, not the general report folder
#         heatmap = generate_gradcam(model, img_tensor, TEST_IMAGE, IMG_OUTPUT_DIR)
#     except Exception as e:
#         print(f"    [CRITICAL FAILURE] Grad-CAM crashed: {e}")
#         import traceback
#         traceback.print_exc()
#         heatmap = np.zeros((224, 224))


#     # 6. LOGIC & REASONING
#     print("[5] Reasoning with Knowledge Graph...")
#     mapper = VisualFindingMapper()
#     findings = mapper.map_findings(heatmap)
#     print(f"    -> Visual Findings: {findings}")

#     reasoner = KGReasoner()
#     diagnoses, confidences, trace = reasoner.reason(
#         findings, 
#         model_probs, 
#         patient_data # <--- Now passing real data
#     )

#     # 7. GENERATE REPORT
#     explainer = ExplanationGenerator()
#     final_report = explainer.generate(
#         findings, 
#         diagnoses, 
#         confidences, 
#         trace, 
#         patient_data
#     )

#     # Save
#     with open(f"{OUTPUT_DIR}/final_diagnosis.txt", "w") as f:
#         f.write(final_report)
    
#     print("\n" + final_report)

# if __name__ == "__main__":
#     main()




import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageStat

# Custom Modules
from src.model_factory import get_model
from src.grad_cam import generate_gradcam
from src.findings_mapper import VisualFindingMapper
from src.kg_reasoner import KGReasoner
from src.explanation_generator import ExplanationGenerator
from src.dataset import LABELS

# ---------------- CONFIGURATION ----------------
MODEL_ARCH = 'densenet121'
MODEL_PATH = "./new_output/models/densenet121_best.pth"
CSV_PATH = "./data/Data_Entry_2017.csv" 
# TEST_IMAGE = "./data/images/00000001_000.png"
# TEST_IMAGE = "./data/normal/NORMAL2.png"
# TEST_IMAGE = "./data/normal/tree.png"
# TEST_IMAGE = "./data/normal/healty.png"
TEST_IMAGE = "./data/images/00000002_000.png"
OUTPUT_DIR = "./new_output/reports"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- GATEKEEPER (CASE 4 FIX) ----------------
def is_valid_xray(img_path):
    """
    Checks if image is a valid X-ray using PIL (No OpenCV needed).
    """
    if not os.path.exists(img_path):
        return False, f"File not found: {img_path}"

    try:
        img = Image.open(img_path).convert('RGB')
        
        # 1. Check Color Saturation (X-rays are Grayscale)
        # Convert to HSV and check 'S' channel
        hsv_img = img.convert('HSV')
        s_channel = np.array(hsv_img)[:, :, 1]
        avg_saturation = s_channel.mean()
        
        # Threshold: > 25 means it has significant color (Car, Cat, etc.)
        if avg_saturation > 25:
            return False, f"Image is too colorful (Sat: {avg_saturation:.1f}). Real X-rays are grayscale."

        # 2. Check for Blank/Black Images (Variance)
        gray_img = img.convert('L')
        stat = ImageStat.Stat(gray_img)
        variance = stat.var[0]
        
        if variance < 10:
            return False, "Image is blank or too blurry."

        return True, "Valid"

    except Exception as e:
        # Ab yeh asli error print karega
        return False, f"Critical Error: {str(e)}"

# ---------------- MAIN PIPELINE ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SYNAPSE PIPELINE ({device}) ---")

    # [STEP 0] GATEKEEPER CHECK
    print(f"[0] Validating Image: {TEST_IMAGE}")
    valid, msg = is_valid_xray(TEST_IMAGE)
    
    if not valid:
        print("\n" + "="*50)
        print(f"🛑 REJECTED (CASE 4 TRIGGERED)")
        print(f"Reason: {msg}")
        print("="*50)
        return

    # [STEP 1] LOAD MODEL
    print(f"[1] Loading Model...")
    try:
        model = get_model(MODEL_ARCH, len(LABELS), device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # [STEP 2] INFERENCE
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(TEST_IMAGE).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    model_probs = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    
    # [STEP 3] VISUALIZATION
    print("[2] Generating Visuals...")
    IMG_OUT = os.path.join(OUTPUT_DIR, "images")
    heatmap = generate_gradcam(model, img_tensor, TEST_IMAGE, IMG_OUT)
    
    mapper = VisualFindingMapper()

    # --- UPDATE: Unpack Findings AND Location ---
    findings, location = mapper.map_findings(heatmap)

    # [STEP 4] LOGIC ENGINE (Handles Case 1, 2, 3)
    print("[3] Reasoning...")
    kg = KGReasoner()
    
    # Fake metadata for demo (Real app would take form input)
    patient_data = {"id": "Unknown", "age": 0, "gender": "?"} 
    
    final_diag, final_conf, trace, status_code = kg.reason(findings, model_probs, patient_data)

    # [STEP 5] REPORT
    explainer = ExplanationGenerator()
    report = explainer.generate(findings, final_diag, final_conf, trace, status_code,location)

    print("\n" + report)
    with open(f"{OUTPUT_DIR}/final_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()