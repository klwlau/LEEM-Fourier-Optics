import matplotlib.pyplot as plt
import numpy as np

def plotArray(plot_data, plotSensitivity=1):
    m, s = np.mean(plot_data), np.std(plot_data)
    plt.imshow(plot_data, interpolation='nearest', cmap='jet',
               vmin=m - plotSensitivity * s, vmax=m + plotSensitivity * s,
               origin='lower')
    plt.colorbar()
    plt.show()