from FO2Dconstants import *

def create2DSimulatedObject():
    amp = 1
    simulatedObject = amp * np.zeros((simulatingSpaceTotalStep,simulatingSpaceTotalStep))

    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = amp

    return simulatedObject


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utilityFunc import *

    plotArray(create2DSimulatedObject())