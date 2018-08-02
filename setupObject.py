from constants import *
from sys import getsizeof

# set up Square Object
K = 1 * np.pi

sampleSpaceTotalStep = 501  # sample size
sampleSpaceSize = 25 * 1e-9  # nm #25
objectSpaceSize = 5 * 1e-9  # nm #5

objectStep = int(objectSpaceSize / sampleSpaceSize * sampleSpaceTotalStep)
sampleCoorXX, sampleCoorYY = np.mgrid[-sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j,
                             -sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j]

sampleStepSize = sampleCoorXX[1][0] - sampleCoorXX[0][0]
sqObject = np.zeros(sampleCoorXX.shape)
sampleCenterX, sampleCenterY = int(sampleSpaceTotalStep / 2 + 1), int(sampleSpaceTotalStep / 2 + 1)
sqObject[sampleCenterX - objectStep:sampleCenterX + objectStep,
sampleCenterY - objectStep:sampleCenterY + objectStep] = 1

objectPhaseShift = K * sqObject

# apply wave function and apply FFT
amp = 1
waveObject = amp * np.exp(1j * objectPhaseShift)
waveObjectFT = np.fft.fftshift(np.fft.fft2(waveObject) / sampleSpaceTotalStep ** 2)


# setup qSpace
qSpaceCoor = 1 / sampleStepSize / (sampleSpaceTotalStep - 1) * np.arange(sampleSpaceTotalStep)
qSpaceCoor = qSpaceCoor - (np.amax(qSpaceCoor) - np.amin(qSpaceCoor)) / 2  # adjust qSpaceCoor center

qSpaceXX, qSpaceYY = np.meshgrid(qSpaceCoor, qSpaceCoor)

# setup aperture function
apertureMask = qSpaceXX ** 2 + qSpaceYY ** 2 <= q_max ** 2
aperture = np.zeros_like(qSpaceYY)
aperture[apertureMask] = 1

# apply aperture function
maskedWaveObjectFT = np.multiply(waveObjectFT, aperture)
maskedQSpaceXX = np.multiply(qSpaceXX, aperture)
maskedQSpaceYY = np.multiply(qSpaceYY, aperture)

testXX = maskedQSpaceXX[sampleCenterX-72:sampleCenterX+72,sampleCenterX-72:sampleCenterX+72]
print(testXX.shape)
testX,testY = np.meshgrid(testXX,testXX)

print(testX.shape)
print(getsizeof(testX)/8/1024/1024)
