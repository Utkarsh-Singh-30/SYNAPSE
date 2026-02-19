import numpy as np

class VisualFindingMapper:
    def map_findings(self, heatmap):
        """
        Returns: 
          1. findings (list): detected patterns e.g. ['Infiltration']
          2. location (str): e.g. "Lower Right Lung Zone"
        """
        findings = []
        mean_val = heatmap.mean()
        max_val = heatmap.max()
        
        # --- 1. DETECT FINDINGS (Standard Logic) ---
        if max_val > 0.80:
            findings.append("Consolidation")
        elif max_val > 0.60:
            findings.append("Infiltration")
        elif mean_val > 0.40:
            findings.append("Opacity")

        # Edges Check (Pneumothorax)
        h, w = heatmap.shape
        border_mean = (heatmap[:10,:].mean() + heatmap[-10:,:].mean() + 
                       heatmap[:,:10].mean() + heatmap[:,-10:].mean()) / 4
        if border_mean > 0.3 and mean_val < 0.2:
            findings.append("Pneumothorax")

        # Center Check (Cardiomegaly)
        center_map = heatmap[int(h/3):int(2*h/3), int(w/3):int(2*w/3)]
        if center_map.mean() > 0.6:
            findings.append("Cardiomegaly")
            
        # Remove duplicates
        findings = list(set(findings))

        # --- 2. DETECT LOCATION (New Logic) ---
        # Find the coordinates of the brightest spot
        y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Vertical Zone
        if y < h * 0.33: vert = "Upper"
        elif y > h * 0.66: vert = "Lower"
        else: vert = "Mid"
        
        # Horizontal Zone
        # Note: Medical convention is patient-centric, but here we describe Image Region
        if x < w * 0.33: horiz = "Right"  # Image Left
        elif x > w * 0.66: horiz = "Left" # Image Right
        else: horiz = "Central"
        
        if horiz == "Central":
            location = "Mediastinal/Cardiac Region"
        else:
            location = f"{horiz} {vert} Lung Zone"

        return findings, location