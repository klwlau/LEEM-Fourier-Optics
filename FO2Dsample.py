from FO2Dconstants import *

def create2DSimulatedObject():
    simulatedObject = np.zeros((simulatingSpaceTotalStep,simulatingSpaceTotalStep))

    amp = 1
    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = amp

    return simulatedObject


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utilityFunc import *

    plotArray(create2DSimulatedObject())