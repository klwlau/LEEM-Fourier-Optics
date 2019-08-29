from constants2DFO import *

def create2DSimulatedObject():
    amp = 1
    simulatedObject = amp * np.zeros((simulatingSpaceTotalStep,simulatingSpaceTotalStep))

    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = 1

    return simulatedObject


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plotArray(plot_data, vmin=-1, vmax=1):
        plt.imshow(plot_data.T, interpolation='nearest', cmap='jet',  # np.flipud(plot_data)
                   origin='lower', vmin=vmin, vmax=vmax)  # vmin=0, vmax=2,
        plt.colorbar()
        plt.show()


    plotArray(create2DSimulatedObject())