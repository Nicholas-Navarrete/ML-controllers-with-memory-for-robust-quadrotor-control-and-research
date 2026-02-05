import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

'''
This program takes multiple CSV files containing success rate vs KF variance data and combines them into a single scatter plot.
The CSV files should be created using 'Control Validation.py'
Nicholas Navarrete
'''

# Directory containing saved CSV results
results_dir = input("Enter the path to the directory containing the success rate CSV files: ").strip()

# Find all CSV files that match the pattern
csv_files = glob.glob(os.path.join(results_dir, "success_rate_vs_kfvariance_*.csv"))

if not csv_files:
    print("ERROR: No CSV files found in 'results/' directory.")
    exit()

plt.figure(figsize=(9, 6))

for file in csv_files:
    try:
        df = pd.read_csv(file)
        if "KF_Variance" not in df.columns or "Success_Rate" not in df.columns:
            print(f"WARNING: Skipping {file}: missing required columns.")
            continue

        model_name = os.path.basename(file).replace("success_rate_vs_kfvariance_", "").replace(".csv", "")
        
        # Plot as points (no lines)
        plt.scatter(
            df["KF_Variance"],
            df["Success_Rate"],
            label=model_name,
            s=40,              # point size
            alpha=0.7          # transparency
        )

    except Exception as e:
        print(f"WARNING: Could not read {file}: {e}")

# Formatting
plt.xscale("log")
plt.xlabel("KF Variance", fontsize=12)
plt.ylabel("Success Rate", fontsize=12)
plt.title("Success Rate vs KF Variance (All Models)", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(title="Model", loc="best", fontsize=10)
plt.tight_layout()

# Save combined plot
combined_plot_filename = os.path.join(results_dir, "combined_success_rate_vs_kfvariance_points.png")
plt.savefig(combined_plot_filename, dpi=300)
plt.show()

print(f"INFO: Combined scatter plot saved to {combined_plot_filename}")
