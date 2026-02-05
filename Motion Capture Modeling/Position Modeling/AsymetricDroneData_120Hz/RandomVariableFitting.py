import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

# Value
DataType = ['x','y','z','roll','pitch','yaw']
# Location

MeasurementLocation = ['x0y0z0','x0y0z1','x0y0z2','x0y1z1','x0y1z2','x1y1z1','x1y1z2','x-0y-1z1']
# Load the data

for Location in MeasurementLocation:
    for Type in DataType:
        filename = Location + '_' + Type + '_buffer.csv'
        df = pd.read_csv(filename)

        # Extract the Type values
        data = df[Type].values

        # Plot the measured points as a histogram
        plt.figure(figsize=(10, 6))
        count, bins, ignored = plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Measured Data')

        # Fit a Gaussian to the data
        mu, std = norm.fit(data)

        # Plot the PDF of the fitted Gaussian
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu$={mu:.5f}, $\sigma$={std:.5f}')

        # Plot the KDE
        sns.kdeplot(data, color='b', label='KDE of Data')

        plt.xlabel(Type)
        plt.ylabel('Probability Density')
        plt.title(Location+' '+Type+' Distribution and Gaussian Fit')
        plt.legend()
        plt.savefig(f'{Location}_{Type}_distribution.png', dpi=300, bbox_inches='tight')