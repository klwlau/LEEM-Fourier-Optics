from scipy import ndimage
import numpy as np

def create2DSimulatedObject(simulatedSpace):
    amp = 1
    simulatedObject = amp * np.zeros_like(simulatedSpace)


    def rippleObject(xCenter, yCenter, xLength, yLength, deg):

        returnMatrix = np.zeros_like(simulatedSpace)

        def rippleFunc(x, y):
            xPixelStart = xCenter - xLength // 2
            xPixelEnd = xCenter + xLength // 2
            yPixelStart = yCenter - yLength // 2
            yPixelEnd = yCenter + yLength // 2

            def mapXY(x, y):
                mapX = (x - xPixelStart) / (xPixelEnd - xPixelStart)
                mapY = (y - yPixelStart) / (yPixelEnd - yPixelStart)
                return mapX, mapY

            mapX, mapY = mapXY(x, y)
            if 0 < mapX < 1 and 0 < mapY < 1:
                return amp * -1 * np.cos(2 * np.pi * mapX) + amp
            else:
                return 0

        def rotateAtCenter(img, angle, pivot):
            pivot[0], pivot[1] = pivot[1], pivot[0]
            # img = np.flipud(img)
            padX = [img.shape[1] - pivot[0], pivot[0]]
            padY = [img.shape[0] - pivot[1], pivot[1]]
            imgP = np.pad(img, [padY, padX], 'constant')
            imgR = ndimage.rotate(imgP, angle, reshape=False)
            return imgR[padY[0]: -padY[1], padX[0]: -padX[1]]

        for x in range(len(returnMatrix)):
            for y in range(len(returnMatrix)):
                returnMatrix[x][y] = rippleFunc(x, y)

        returnMatrix = rotateAtCenter(returnMatrix, deg, [xCenter, yCenter])

        return returnMatrix

    simulatedObject[251 - 50:251 + 50, 251 - 50:251 + 50] = 1

    return simulatedObject