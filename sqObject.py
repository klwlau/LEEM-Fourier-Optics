# Square Object
K = 1 * np.pi

sampleSpaceTotalStep = 501  # sample size
sampleSpaceSize = 25 * 1e-9  # nm #25
objectSpaceSize = 5 * 1e-9  # nm #5

objectStep = int(objectSpaceSize / sampleSpaceSize * sampleSpaceTotalStep)
sampleCoorXX, sampleCoorYY = np.mgrid[-sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j,
                             -sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j]

sqObject = np.zeros(sampleCoorXX.shape)
sampleCenterX, sampleCenterY = int(sampleSpaceTotalStep / 2 + 1), int(sampleSpaceTotalStep / 2 + 1)
sqObject[sampleCenterX - objectStep:sampleCenterX + objectStep,
sampleCenterY - objectStep:sampleCenterY + objectStep] = 1

objectPhaseShift = K * sqObject

amp = 1
wave_obj = amp * np.exp(1j * objectPhaseShift)
F_wave_obj = np.fft.fft2(wave_obj) / sampleSpaceTotalStep ** 2

# q = 1/(l(2,1)-l(1,1))*(0:1:n-1)/(sampleSpaceTotalStep-1)
#
# F_wave_obj = fftshift(F_wave_obj)
# q = q-(max(q)-min(q))/2
# q = transpose(q)
# [qx,qy] = meshgrid(q)
#
# mask = double(qx.**2+qy.**2 <= q_max**2)
# mask_F_wave_obj = F_wave_obj(mask == 1)
# mask_qx = qx(mask == 1)
# mask_qy = qy(mask ==1)