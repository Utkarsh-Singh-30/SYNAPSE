import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIGURATION ----------------
REPORT_DIR = './new_output/final_reports'
PLOT_DIR = './new_output/final_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

def parse_reports():
    """Reads text reports and extracts metrics."""
    files = glob.glob(os.path.join(REPORT_DIR, "*_detailed_report.txt"))
    if not files:
        files = glob.glob("*_detailed_report.txt")
        
    if not files:
        print(f"[ERROR] No report files found.")
        return pd.DataFrame()

    data = []
    print(f"Found {len(files)} reports. Parsing...")

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Default values
            metrics = {'Model': 'Unknown', 'AUC': 0.0, 'Accuracy': 0.0, 
                       'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
            
            for line in lines:
                line = line.strip()
                if line.startswith("MODEL ARCHITECTURE:"):
                    metrics['Model'] = line.split(":", 1)[1].strip()
                elif line.startswith("OVERALL VALIDATION AUC:"):
                    metrics['AUC'] = float(line.split(":", 1)[1].strip())
                elif line.startswith("OVERALL ACCURACY:"):
                    metrics['Accuracy'] = float(line.split(":", 1)[1].strip())
                elif line.startswith("OVERALL RECALL:"):
                    metrics['Recall'] = float(line.split(":", 1)[1].strip())
                elif line.startswith("weighted avg"):
                    parts = line.split()
                    if len(parts) >= 5:
                        metrics['Precision'] = float(parts[2])
                        metrics['F1'] = float(parts[4])
                        if metrics['Recall'] == 0.0: # Fallback
                            metrics['Recall'] = float(parts[3])

            # =======================================================
            #  CUSTOM RENAMING LOGIC (CHANGE X-AXIS NAMES HERE)
            # =======================================================
            # 1. Clean up common suffixes
            raw_name = metrics['Model'].replace("_best", "").lower()

            if "ensemble" in raw_name:
                metrics['Model'] = "Ensemble_Model"  # New Name
            elif "densenet121" in raw_name:
                metrics['Model'] = "DenseNet-121"    # New Name
            elif "seresnext50" in raw_name:
                metrics['Model'] = "SE-ResNeXt-50"   # New Name
            elif "resnet101" in raw_name:
                metrics['Model'] = "ResNet-101"      # New Name
            elif "thoraxnet" in raw_name:
                metrics['Model'] = "ThoraxNet"       # New Name
            # =======================================================
            
            data.append(metrics)
            print(f"  -> Parsed {metrics['Model']}")

        except Exception as e:
            print(f"  [ERROR] Failed to parse {file_path}: {e}")

    return pd.DataFrame(data)

def plot_single_metric(df, metric, title, filename, color):
    """Generates a clean Line Graph for a single metric."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sort for cleaner line
    df_sorted = df.sort_values(by=metric)
    
    sns.lineplot(data=df_sorted, x='Model', y=metric, marker='o', 
                 linewidth=3, markersize=10, color=color)
    
    # Add Text Labels
    for i in range(df_sorted.shape[0]):
        x = df_sorted.iloc[i]['Model']
        y = df_sorted.iloc[i][metric]
        plt.text(x, y + 0.002, f"{y:.4f}", ha='center', va='bottom', 
                 fontweight='bold', color='black', fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"[SAVED] {title} -> {save_path}")

def plot_multi_metric(df):
    """Generates the Multi-Line Chart with 5 Metrics."""
    df_melted = df.melt(id_vars=['Model'], 
                        value_vars=['AUC', 'Accuracy', 'Precision', 'Recall', 'F1'], 
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    sns.lineplot(data=df_melted, x='Model', y='Score', hue='Metric', 
                 style='Metric', markers=True, dashes=False, linewidth=2.5, markersize=9)

    for i in range(df_melted.shape[0]):
        x = df_melted.iloc[i]['Model']
        y = df_melted.iloc[i]['Score']
        val = f"{y:.2f}"
        plt.text(x, y + 0.01, val, ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.title("Comprehensive Performance (5 Metrics)", fontsize=16, fontweight='bold')
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "multi_metric_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"[SAVED] Multi-Metric Chart -> {save_path}")

if __name__ == "__main__":
    df = parse_reports()
    if not df.empty:
        # Plot 1: AUC
        plot_single_metric(df, 'AUC', "Model vs AUC Score", "model_vs_auc.png", "#1f77b4")
        # Plot 2: Recall
        plot_single_metric(df, 'Recall', "Model vs Recall (Sensitivity)", "model_vs_recall.png", "#2ca02c")
        # Plot 3: All Metrics
        plot_multi_metric(df)