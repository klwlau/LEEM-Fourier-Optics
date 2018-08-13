import matplotlib.pyplot as plt
import numpy as np

def plotArray(plot_data, plotSensitivity=1):
    plt.imshow(plot_data, interpolation='nearest', cmap='jet',
               origin='lower') #vmin=0, vmax=2,
    plt.colorbar()
    plt.show()