class ExplanationGenerator:
    def generate(self, findings, diagnoses, confidence, trace, status_code, location=None):
        
        report = []
        report.append("="*60)
        report.append(f"{'SYNAPSE DIAGNOSTIC REPORT':^60}")
        report.append("="*60 + "\n")

        # --- CASE 3: NORMAL ---
        if status_code == 3:
            report.append("[RESULT]: NO DISEASE FOUND (NORMAL)")
            report.append("-" * 40)
            report.append("Explanation: The analysis indicates a healthy X-ray.")
            report.append("Reasoning: No significant pathological patterns were detected.")
            report.append("\nRecommendation: Routine checkup / No action required.")
            report.append("="*60)
            return "\n".join(report)

        # --- CASE 2: UNSPECIFIED ---
        if status_code == 2:
            suspected = diagnoses[0]
            report.append("[RESULT]: UNSPECIFIED ABNORMALITY DETECTED")
            report.append("-" * 40)
            report.append(f"Observation: System detected potential anomaly ({suspected}) but visual patterns are ambiguous.")
            report.append("Explanation: This finding does not perfectly match the standard 14 disease profiles.")
            report.append("\nRecommendation: DOCTOR REVIEW REQUIRED.")
            report.append("="*60)
            return "\n".join(report)

        # --- CASE 1: DISEASE DETECTED (The Update) ---
        if status_code == 1:
            primary = diagnoses[0]
            score = confidence[primary] * 100
            
            report.append(f"[RESULT]: {primary.upper()} DETECTED")
            report.append("-" * 40) 
            
            # --- NEW EXPLANATION LOGIC ---
            finding_text = ", ".join(findings) if findings else "clinical patterns"
            location_text = f" in the {location}" if location else ""
            
            report.append(
                f"Explanation: There is a {score:.1f}% chance of {primary} because the X-ray shows "
                f"{finding_text}{location_text}."
            )
            
            # Top 3 Table
            report.append("\n[TOP 3 PROBABILITIES]")
            sorted_conf = sorted(confidence.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (d, p) in enumerate(sorted_conf):
                report.append(f"  {i+1}. {d:<20} : {p*100:.1f}%")

            report.append("\nRecommendation: Please consult a Pulmonologist for confirmation.")
            report.append("="*60)
            return "\n".join(report)

        return "Report Generation Error."