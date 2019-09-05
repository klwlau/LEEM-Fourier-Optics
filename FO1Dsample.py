from FO1Dconstants import *


def create2DSimulatedObject():
    object_wavelength = 900e-9
    n_sample = 1 + 2 ^ 10
    simulatedObject = np.linspace(-object_wavelength, object_wavelength, n_sample)

    K = [3, 4, 5, 6] * 2
    n_max = np.floor(q_max / (1 / object_wavelength))
    q = (1 / object_wavelength) * np.arange(-n_max, n_max, 1)

    return simulatedObject


if __name__ == '__main__':
    from utilityFunc import *

    plot1DArray(create2DSimulatedObject())
