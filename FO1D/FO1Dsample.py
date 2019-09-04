from FO1Dconstants import *

def create2DSimulatedObject():
    t=100
    fs=1000
    samples = np.linspace(0, t, t * fs)
    simulatedObject = np.sin(samples)


    return simulatedObject


if __name__ == '__main__':
    from utilityFunc import *

    plot1DArray(create2DSimulatedObject())