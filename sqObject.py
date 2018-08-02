from constants import *

# Square Object
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

amp = 1
waveObject = amp * np.exp(1j * objectPhaseShift)
waveObjectFT = np.fft.fftshift(np.fft.fft2(waveObject) / sampleSpaceTotalStep ** 2)

qSpaceCoor = 1 / sampleStepSize / (sampleSpaceTotalStep - 1) * np.arange(sampleSpaceTotalStep)
qSpaceCoor = qSpaceCoor - (np.amax(qSpaceCoor) - np.amin(qSpaceCoor)) / 2  # adjust qSpaceCoor center

qSpaceXX, qSpaceYY = np.meshgrid(qSpaceCoor, qSpaceCoor)

apertureMask = qSpaceXX ** 2 + qSpaceYY ** 2 <= q_max ** 2
aperture = np.zeros_like(qSpaceYY)
print(aperture.shape)
aperture[apertureMask] = 1
print(aperture.shape)

maskedWaveObjectFT = np.multiply(waveObjectFT, aperture)
maskedQSpaceXX = np.multiply(qSpaceXX, aperture)
maskedQSpaceYY = np.multiply(qSpaceYY, aperture)

# mask_F_wave_obj = F_wave_obj(mask == 1)
# mask_qx = qx(mask == 1)
# mask_qy = qy(mask ==1)
