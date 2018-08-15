import numpy as np
import matplotlib.pyplot as plt

simulatedSpace = np.zeros((501,501))

def createSimulatedObject():
    amp = 1

    def rippleObject(xPixelStart, xPixelEnd, yPixelStart, yPixelEnd, deg):
        matrix = np.zeros_like(simulatedSpace)
        radTheta = np.radians(deg)

        def rippleFunc(x):  # , y):
            mapX = (x - xPixelStart) / (xPixelEnd - xPixelStart)
            # mapY = (y-yPixelStart) / (yPixelEnd - yPixelStart)
            if 1>mapX>0:
                print(mapX)
                return amp * -1 * np.cos(2 * np.pi * mapX)
            else:
                return 0

        for x in range(len(matrix)):
            for y in range(len(matrix)):
                mapX = x * np.cos(radTheta) - y * np.sin(radTheta)
                mapY = x * np.sin(radTheta) + y * np.cos(radTheta)
                # matrix[x][y] = rippleFunc(mapX, mapY)

        testX = np.linspace(0, 501, 500)
        testY = np.zeros_like(testX)
        for i in range(len(testY)):
            # print(i)
            testY[i] = rippleFunc(i)
        plt.plot(testX, testY)
        plt.show()

    rippleObject(100,300,100,300,0)

createSimulatedObject()