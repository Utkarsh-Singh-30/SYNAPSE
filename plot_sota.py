import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ---------------- CONFIGURATION ----------------
PLOT_DIR = './new_output/final_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- SOTA DATA (OFFICIAL SCORES) ----------------
# You can change the 'Model' names below to whatever you want on the X-axis.
sota_data = [
    {'Model': 'Wang et al.\n(2017)',   'AUC': 0.738, 'Type': 'Previous Work'},
    {'Model': 'Yao et al.\n(2018)',    'AUC': 0.798, 'Type': 'Previous Work'},
    {'Model': 'CheXNet\n(Stanford)',   'AUC': 0.841, 'Type': 'Previous Work'},
    {'Model': 'Our Ensemble\n(Final)', 'AUC': 0.857, 'Type': 'Our Work'} 
]

# sota_data = [
#     {'Model': 'CheXNet\n(2017)',          'AUC': 0.841, 'Type': 'Previous Work'},
#     {'Model': 'Swin Trans. V2\n(2022)',   'AUC': 0.854, 'Type': 'Previous Work'},
#     {'Model': 'ConvNeXt V2\n(2023)',      'AUC': 0.855, 'Type': 'Previous Work'},
#     {'Model': 'Foundation Model\n(2023)', 'AUC': 0.861, 'Type': 'Previous Work'}, # Google ELIXR
#     {'Model': 'Our Ensemble\n(Proposed)', 'AUC': 0.857, 'Type': 'Our Work'} 
# ]

def plot_sota_comparison():
    df = pd.DataFrame(sota_data)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Define Colors: Grey for others, Red for your model
    colors = {'Previous Work': 'grey', 'Our Work': '#d62728'}
    
    ax = sns.barplot(data=df, x='Model', y='AUC', hue='Type', 
                     palette=colors, dodge=False)

    # Add Score Numbers on Top of Bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.005, 
                    f'{height:.3f}', 
                    ha="center", va='bottom', fontsize=12, fontweight='bold', color='black')

    plt.title("Comparison with State-of-the-Art (NIH ChestX-ray14)", fontsize=16, fontweight='bold')
    plt.ylabel("AUC Score", fontsize=14)
    plt.xlabel("Research Model", fontsize=14)
    plt.ylim(0.7, 0.9) # Zoom focus
    plt.legend(title=None, loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "sota_comparison_chart1.png")
    plt.savefig(save_path, dpi=300)
    print(f"[SAVED] SOTA Comparison -> {save_path}")

if __name__ == "__main__":
    plot_sota_comparison()