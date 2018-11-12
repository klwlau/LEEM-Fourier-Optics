import matplotlib.pyplot as plt
import numpy as np

def plotArray(plot_data, vmin = -1,vmax=1):
    plt.imshow(plot_data.T, interpolation='nearest', cmap='jet', #np.flipud(plot_data)
               origin='lower',vmin=vmin, vmax=vmax) #vmin=0, vmax=2,
    plt.colorbar()
    plt.show()