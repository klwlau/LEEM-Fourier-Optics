import matplotlib.pyplot as plt
import numpy as np

def plotArray(plot_data, plotSensitivity=1):
    plt.imshow(plot_data.T, interpolation='nearest', cmap='jet', #np.flipud(plot_data)
               origin='lower') #vmin=0, vmax=2,
    plt.colorbar()
    plt.show()