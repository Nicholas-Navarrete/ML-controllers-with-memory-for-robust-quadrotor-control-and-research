import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

# Load the frequency file
df = pd.read_csv("Frequencies 1.csv")

# Extract frequency data
data = df["frequency_hz"].values
n = len(data)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram
ax.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Measured Data')

# Gaussian fit
mu, std = norm.fit(data)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, mu, std)

ax.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\nμ={mu:.5f}, σ={std:.5f}')

# KDE
sns.kdeplot(data, color='b', label='KDE of Data', ax=ax)

# Labels and title
ax.set_title(f"Frequency Distribution (n={n})")
ax.set_xlabel("frequency_hz")
ax.set_ylabel("Probability Density")
ax.legend()

plt.tight_layout()
plt.savefig("Frequency_Distribution_2.png", dpi=300, bbox_inches='tight')
plt.close()
