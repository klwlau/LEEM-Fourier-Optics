from joblib import Parallel, delayed
import multiprocessing
# import matplotlib.pyplot as plt
#####import constants######
from constants import *


######set up Square Object#######
K = 1 * np.pi

sampleSpaceTotalStep = 501  # sample size
sampleSpaceSize = 25 * 1e-9  # nm #25
objectSpaceSize = 5 * 1e-9  # nm #5

objectStep = int(objectSpaceSize / sampleSpaceSize * sampleSpaceTotalStep)
sampleCoorRealSpaceXX, sampleCoorRealSpaceYY = np.mgrid[-sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j,
                                               -sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j]

sampleStepSize = sampleCoorRealSpaceXX[1][0] - sampleCoorRealSpaceXX[0][0]
sqObject = np.zeros(sampleCoorRealSpaceXX.shape)
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
maskedWaveObjectFT = waveObjectFT[aperture == 1]

maskedQSpaceXX = qSpaceXX[aperture == 1]
maskedQSpaceYY = qSpaceYY[aperture == 1]

print("making transmittion CrossCoefficientMatrix")

Qx_i, Qx_j = np.meshgrid(maskedQSpaceXX, maskedQSpaceXX, sparse=True)
Qy_i, Qy_j = np.meshgrid(maskedQSpaceYY, maskedQSpaceYY, sparse=True)
F_i, F_j = np.meshgrid(maskedWaveObjectFT, maskedWaveObjectFT, sparse=True)

Qi = Qx_i + 1j * Qy_i
Qj = Qx_j + 1j * Qy_j

abs_Qi = (Qx_i ** 2 + Qy_i ** 2) ** 0.5
abs_Qj = (Qx_j ** 2 + Qy_j ** 2) ** 0.5

print("cal abs_Qi,abs_Qj power")
abs_Qi_2 = abs_Qi ** 2
abs_Qi_4 = abs_Qi_2 ** 2
abs_Qi_6 = abs_Qi_2 ** 3
abs_Qj_2 = abs_Qj ** 2
abs_Qj_4 = abs_Qj_2 ** 2
abs_Qj_6 = abs_Qj_2 ** 3

print("calc T_o")
T_o = np.exp(1j * 2 * np.pi * (1 / 4 * C_3 * lamda ** 3 * (abs_Qi_4 - abs_Qj_4)
                               + 1 / 6 * C_5 * lamda ** 5 * (abs_Qi_6 - abs_Qj_6)
                               - 1 / 2 * delta_z * lamda * (abs_Qi_2 - abs_Qj_2)
                               ))
print("calc E_s")
E_s = np.exp(-np.pi ** 2 / 4 / np.log(2) * q_ill ** 2 *
             np.abs(C_3 * lamda ** 3 * (Qi * abs_Qi_2 - Qj * abs_Qj_2)
                    + C_5 * lamda ** 5 * (Qi * abs_Qi_4 - Qj * abs_Qj_4)
                    - delta_z * lamda * (Qi - Qj)) ** 2)

print("calc E_cc")

E_cc = (1 - 1j * np.pi / 4 / np.log(2) *
        delta_fcc * lamda * (abs_Qi_2 - abs_Qj_2)) ** -0.5

print("calc E_ct")
E_ct = E_cc * np.exp(-np.pi ** 2 / 16 / np.log(2) *
                     ((delta_fc * lamda) * (abs_Qi_2 - abs_Qj_2)
                      + (1 / 2 * delta_f3c * lamda ** 3) *
                      (abs_Qi_4 - abs_Qj_4)) ** 2 * E_cc ** 2)

print("calc T")
T = T_o * E_s * E_ct


##############cal Matrix I##########

def calI(element, TElement):
    return element * TElement * EXP


qq_i = maskedQSpaceXX + maskedQSpaceYY * 1j
qq_j = maskedQSpaceXX + maskedQSpaceYY * 1j

EXP = np.exp(1j * 2 * np.pi * (np.sum((qq_i - qq_j[:, np.newaxis]).real) * sampleCoorRealSpaceXX
                               + np.sum((qq_i - qq_j[:, np.newaxis]).imag) * sampleCoorRealSpaceYY))

abs_maskedWaveObjectFTRavel = (maskedWaveObjectFT * np.conj(maskedWaveObjectFT[:, np.newaxis])).ravel()
TRavel = T.ravel()

print(EXP.shape)
print(T.ravel().shape)
print(abs_maskedWaveObjectFTRavel.shape)

num_cores = multiprocessing.cpu_count()
print("Start multiprocessing")

# multicoreResults = Parallel(n_jobs=num_cores)(
#     delayed(calI)(element, TElement) for element, TElement in zip(abs_maskedWaveObjectFTRavel, TRavel))
counter = 0
multicoreResults = np.zeros_like(calI(abs_maskedWaveObjectFTRavel[0],TRavel[0]))
for i,j in zip(abs_maskedWaveObjectFTRavel, TRavel):
    multicoreResults += calI(i,j)
    print(counter)
    counter+=1


print("End multiprocessing")

multicoreResults = np.array(multicoreResults)
matrixI = multicoreResults
# matrixI = np.sum(multicoreResults, axis=0)

matrixI = np.fft.fftshift(matrixI)
matrixI = np.absolute(matrixI)

print("End Main")
print(matrixI.shape)

# plt.imshow(matrixI)
# plt.show()

