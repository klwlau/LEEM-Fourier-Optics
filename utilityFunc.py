import matplotlib.pyplot as plt
import numpy as np

def plot2DArray(plot_data, vmin = None, vmax=None):
    plt.imshow(plot_data.T, interpolation='nearest', #np.flipud(plot_data), cmap='jet'
               origin='lower',vmin=vmin, vmax=vmax) #vmin=0, vmax=2,
    plt.colorbar()
    plt.show()


def plot1DArray(plot_data):
    plt.plot(plot_data)
    plt.show()