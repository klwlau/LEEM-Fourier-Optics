from FO2Dconstants import *


def create2DSimulatedObject():
    # define a zero array with dimension 501 by 501.
    simulatedObject = np.zeros((501, 501))

    amp = 1
    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = amp

    return simulatedObject


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utilityFunc import *

    plot2DArray(create2DSimulatedObject())
