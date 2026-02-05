import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

DataType = ['x','y','z','roll','pitch','yaw']
MeasurementLocation = ['x0y0z0','x0y0z1','x0y0z2','x0y1z1','x0y1z2','x1y1z1','x1y1z2','x-0y-1z1']

for Type in DataType:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()
    for idx, Location in enumerate(MeasurementLocation):
        filename = f"{Location}_{Type}_buffer.csv"
        df = pd.read_csv(filename)
        data = df[Type].values

        ax = axes[idx]
        ax.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Measured Data')
        mu, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu$={mu:.5f}, $\sigma$={std:.5f}')
        sns.kdeplot(data, color='b', label='KDE of Data', ax=ax)
        ax.set_title(Location)
        ax.set_xlabel(Type)
        if idx % 4 == 0:
            ax.set_ylabel('Probability Density')
        ax.legend()
    # Hide the unused subplot (last one)
    if len(MeasurementLocation) < len(axes):
        for j in range(len(MeasurementLocation), len(axes)):
            fig.delaxes(axes[j])
    plt.suptitle(f'{Type} Distribution Across Locations')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{Type}_across_locations.png', dpi=300, bbox_inches='tight')
    plt.close()