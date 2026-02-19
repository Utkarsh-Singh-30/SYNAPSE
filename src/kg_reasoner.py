import json
import os

KG_PATH = './data/knowledge_graph/graph_data.json'

class KGReasoner:
    def __init__(self):
        self.visual_rules = {}
        # Load Knowledge Graph
        if os.path.exists(KG_PATH):
            with open(KG_PATH, 'r') as f:
                data = json.load(f)
                self.visual_rules = data.get("visual_rules", {})
                
    def _check_metadata_rules(self, disease, age, gender):
        """
        Returns: (is_consistent, message)
        Checks if the disease makes sense for the patient's Age/Gender.
        """
        msg = []
        is_consistent = True

        # --- RULE 1: AGE RISK FACTORS ---
        # Pneumonia & Hernia are riskier in older patients
        if disease in ['Pneumonia', 'Hernia', 'Emphysema'] and age > 60:
            msg.append(f"Patient age ({age}) puts them in High Risk group for {disease}.")
        
        # Emphysema is very rare in children (usually smoker's disease)
        if disease == 'Emphysema' and 0 < age < 18:
            msg.append(f"⚠️ Unusual finding: {disease} is rare in patients under 18.")
            is_consistent = False # Flag as suspicious

        # --- RULE 2: GENDER SPECIFICS ---
        # Certain types of Hernia are slightly more common in older females
        if disease == 'Hernia' and gender == 'F' and age > 50:
            msg.append(f"Demographics (Female, >50) align with higher risk of Hiatal Hernia.")

        return is_consistent, msg

    def reason(self, findings, model_probs, patient_metadata=None):
        """
        Output: (diagnoses, confidence, trace, STATUS_CODE)
        STATUS_CODE:
          1 = Disease Detected (Confirmed)
          2 = Ambiguous (Visuals or Metadata mismatch)
          3 = Normal (Healthy)
        """
        # 1. Sort Probabilities (High to Low)
        sorted_probs = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)
        top_disease, top_score = sorted_probs[0]
        
        trace = []
        final_confs = {}
        
        # Extract Patient Data (Safely)
        age = 0
        gender = "Unknown"
        if patient_metadata:
            age = patient_metadata.get('age', 0)
            gender = patient_metadata.get('gender', 'Unknown')

        # --- CASE 3: NORMAL CHECK ---
        # Agar Model ka confidence bohot kam hai (< 0.45)
        if top_score < 0.45:
            return ["NORMAL"], {"NORMAL": 1.0 - top_score}, ["All probabilities are low (Healthy Scan)."], 3

        # --- STEP A: CHECK VISUAL EVIDENCE (Grad-CAM) ---
        visual_support = False
        if top_disease in self.visual_rules:
            # Check if findings (e.g., 'Lower Lung') match rules
            matches = [f for f in findings if f in self.visual_rules[top_disease]] 
            if matches:
                visual_support = True
                trace.append(f"✅ Visual evidence ({', '.join(matches)}) matches {top_disease} location.")
            else:
                trace.append(f"❓ Visual findings {findings} do not match typical {top_disease} location.")
        else:
            trace.append("No specific visual rules found for this disease.")

        # --- STEP B: CHECK METADATA (Neuro-Symbolic Logic) ---
        meta_consistent = True
        if age > 0: # Sirf tab check karo agar Age valid hai
            meta_consistent, meta_msgs = self._check_metadata_rules(top_disease, age, gender)
            trace.extend(meta_msgs)
        
        # --- FINAL DECISION LOGIC ---
        
        # Collect Top 3 scores for report
        for d, p in sorted_probs[:3]:
            final_confs[d] = p

        # DECISION 1: UNSPECIFIED / AMBIGUOUS (Case 2)
        # Agar Visuals match nahi kar rahe YA Metadata bohot ajeeb hai (e.g. Kid with Emphysema)
        if not visual_support or not meta_consistent:
            trace.append(f"⚠️ Alert: Model predicts {top_disease} but evidence is inconsistent.")
            return [top_disease], {top_disease: top_score}, trace, 2

        # DECISION 2: CONFIRMED DISEASE (Case 1)
        # Visuals bhi sahi hain, aur Metadata bhi rok nahi raha
        trace.append(f"🏆 Diagnosis {top_disease} is clinically & visually confirmed.")
        return [top_disease], final_confs, trace, 1