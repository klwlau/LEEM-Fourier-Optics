from joblib import Parallel, delayed
import multiprocessing
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


##############cal Matrix I##########

def calI(element, TElement):
    return element * TElement * EXP


qq_i = maskedQSpaceXX + maskedQSpaceYY * 1j
qq_j = maskedQSpaceXX + maskedQSpaceYY * 1j

RoConstant0 = 1j * 2 * np.pi()
RoConstant1 = 1 / 4 * C_3 * lamda ** 3
RoConstant2 = 1 / 6 * C_5 * lamda ** 5
RoConstant3 = -1/2*delta_z*lamda

EsConstant0 = -np.pi**2/4/np.log(2)*q_ill**2
EsConstant1 = C_3*lamda**3
EsConstant2 = C_5*lamda**5
EsConstant3 = - delta_z*lamda

EccConstant0 = 1j*np.pi/4/np.log(2)*delta_fcc*lamda



for i in range(len(maskedQSpaceXX)):

    qq_i = maskedQSpaceXX(i) + 1j * maskedQSpaceYY(i)
    abs_qq_i = np.absolute(qq_i)

    abs_qq_i_2 = abs_qq_i**2
    abs_qq_i_4 = abs_qq_i_2 ** 2
    abs_qq_i_6 = abs_qq_i_2 **3

    for j in range(len(maskedQSpaceYY)):
        qq_j = maskedQSpaceXX(j) + 1j * maskedQSpaceYY(j)
        abs_qq_j = np.absolute(qq_j)

        abs_qq_j_2 = abs_qq_j ** 2
        abs_qq_j_4 = abs_qq_j_2 ** 2
        abs_qq_j_6 = abs_qq_j_2 ** 3

        R_o = np.exp(RoConstant0*
            (RoConstant1 * (abs_qq_i_4 - abs_qq_j_4)
            +RoConstant2 * (abs_qq_i_6 - abs_qq_j_6)
            +RoConstant3 * (abs_qq_i_2 - abs_qq_j_2))
            )
        E_s = np.exp(EsConstant0*
            np.abs(EsConstant1 * (qq_i*abs_qq_i_2 - qq_j*abs_qq_j_2)
            + EsConstant2 * (qq_i*abs_qq_i_4 - qq_j*abs_qq_j_4)
            +EsConstant3 * (qq_i - qq_j))**2
            )
        E_cc = np.sqrt(1 - EccConstant0* (abs_qq_i_2 - abs_qq_j_2))













print("Start multiprocessing")
# multicoreResults = Parallel(n_jobs=num_cores)(
#     delayed(calI)(element, TElement) for element, TElement in zip(abs_maskedWaveObjectFTRavel, TRavel))
# multicoreResults = np.array(multicoreResults)
# matrixI = np.sum(multicoreResults, axis=0)
print("End multiprocessing")

# matrixI = multicoreResults
#
# matrixI = np.fft.fftshift(matrixI)
# matrixI = np.absolute(matrixI)

print("End Main")
