from FO2Dconstants import *

def create2DSimulatedObject():
    # define a zero array with dimension simulatingSpaceTotalStep by simulatingSpaceTotalStep.
    # simulatingSpaceTotalStep definition is load by "from FO2Dconstants import *"
    simulatedObject = np.zeros((simulatingSpaceTotalStep,simulatingSpaceTotalStep))

    amp = 1
    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = amp

    return simulatedObject


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utilityFunc import *

    plotArray(create2DSimulatedObject())