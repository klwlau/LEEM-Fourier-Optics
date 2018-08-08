print("Main Started, Loading Libraries")
import time
from datetime import datetime
import pytz
from joblib import Parallel, delayed
import multiprocessing
from constants import *
import numexpr as ne

fmt = '%d/%m %H:%M:%S'
hkTimeZone = pytz.timezone('Asia/Hong_Kong')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    start_time = time.time()

    def printStatus(counter, done=False, loopMode=False):
        if counter != 0:
            elapsedTime = ((time.time() - start_time) / 60)
            progress = (counter / len(maskedQSpaceXX)) * 100
            totalTime = elapsedTime / (progress / 100)
            timeLeft = totalTime - elapsedTime
            hkDT = datetime.now(hkTimeZone)
            currentHKTime = hkDT.strftime(fmt)
            if done:
                print("-Total Time: %.2f Minutes -" % elapsedTime)
            else:
                if loopMode:
                    print("Loop:", loopMainCounter, "/", loopLen,
                          "-ID:" + str(counter) + "--Elapsed Time: %.2f / %.2f Minutes -" % (elapsedTime, totalTime)
                          + "Time Left: %.2f  Minutes -" % timeLeft + "%.2f" % progress + "%-" + currentHKTime)
                else:
                    print("-ID:" + str(counter) + "--Elapsed Time: %.2f / %.2f Minutes -" % (elapsedTime, totalTime)
                          + "Time Left: %.2f  Minutes -" % timeLeft + "%.2f" % progress + "%-" + currentHKTime)

    ######set up Square Object#######
    K = 1 * np.pi
    q_max = alpha_ap / lamda
    q_ill = alpha_ill / lamda

    sampleSpaceTotalStep = 501  # sample size
    sampleSpaceSize = 25 * 1e-9  # nm #25
    objectSpaceSize = 5 * 1e-9  # nm #5

    objectStep = int((objectSpaceSize / sampleSpaceSize * sampleSpaceTotalStep) / 2)
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

    # Qx_i, Qx_j = np.meshgrid(maskedQSpaceXX, maskedQSpaceXX, sparse=True)
    # Qy_i, Qy_j = np.meshgrid(maskedQSpaceYY, maskedQSpaceYY, sparse=True)
    # F_i, F_j = np.meshgrid(maskedWaveObjectFT, maskedWaveObjectFT, sparse=True)

    ##############cal Matrix I##########

    # qq_i = maskedQSpaceXX + maskedQSpaceYY * 1j
    # qq_j = maskedQSpaceXX + maskedQSpaceYY * 1j

    RoConstant0 = 1j * 2 * np.pi
    RoConstant1 = 1 / 4 * C_3 * lamda ** 3
    RoConstant2 = 1 / 6 * C_5 * lamda ** 5
    RoConstant3 = -1 / 2 * delta_z * lamda

    EsConstant0 = -np.pi ** 2 / 4 / np.log(2) * q_ill ** 2
    EsConstant1 = C_3 * lamda ** 3
    EsConstant2 = C_5 * lamda ** 5
    EsConstant3 = - delta_z * lamda

    EccConstant0 = 1j * np.pi / 4 / np.log(2) * delta_fcc * lamda

    EctConstant0 = -np.pi ** 2 / 16 / np.log(2)
    EctConstant1 = delta_fc * lamda
    EctConstant2 = 1 / 2 * delta_f3c * lamda ** 3

    def outerForLoop(counter_i,sampleCoorRealSpaceXX = sampleCoorRealSpaceXX,sampleCoorRealSpaceYY=sampleCoorRealSpaceYY):
        # global returnMatrix
        returnMatrix = np.zeros_like(sampleCoorRealSpaceXX)
        qq_i = maskedQSpaceXX[counter_i] + 1j * maskedQSpaceYY[counter_i]
        abs_qq_i = np.absolute(qq_i)

        abs_qq_i_2 = abs_qq_i ** 2
        abs_qq_i_4 = abs_qq_i_2 ** 2
        abs_qq_i_6 = abs_qq_i_2 ** 3

        for counter_j in range(len(maskedQSpaceYY)):
            qq_j = maskedQSpaceXX[counter_j] + 1j * maskedQSpaceYY[counter_j]
            abs_qq_j = np.absolute(qq_j)

            abs_qq_j_2 = abs_qq_j ** 2
            abs_qq_j_4 = abs_qq_j_2 ** 2
            abs_qq_j_6 = abs_qq_j_2 ** 3

            R_o = np.exp(RoConstant0 *
                         (RoConstant1 * (abs_qq_i_4 - abs_qq_j_4)
                          + RoConstant2 * (abs_qq_i_6 - abs_qq_j_6)
                          + RoConstant3 * (abs_qq_i_2 - abs_qq_j_2))
                         )
            E_s = np.exp(EsConstant0 *
                         np.abs(EsConstant1 * (qq_i * abs_qq_i_2 - qq_j * abs_qq_j_2)
                                + EsConstant2 * (qq_i * abs_qq_i_4 - qq_j * abs_qq_j_4)
                                + EsConstant3 * (qq_i - qq_j)) ** 2
                         )
            E_cc = np.sqrt(1 - EccConstant0 * (abs_qq_i_2 - abs_qq_j_2))

            E_ct = E_cc * np.exp(EctConstant0 *
                                 (EctConstant1 * (abs_qq_i_2 - abs_qq_j_2)
                                  + EctConstant2 * (abs_qq_i_4 - abs_qq_j_4)) ** 2 * E_cc ** 2)

            temp = 2j * np.pi * ((qq_i - qq_j).real * sampleCoorRealSpaceXX + (qq_i - qq_j).imag * sampleCoorRealSpaceYY)
            # temp = temp.astype("Float32")

            EXP = ne.evaluate("exp(temp)")

            returnMatrix = returnMatrix + R_o * E_s * E_ct * maskedWaveObjectFT[counter_i] * np.conj(
                maskedWaveObjectFT[counter_j]) * EXP

        return returnMatrix

    num_cores = multiprocessing.cpu_count()-4
    totalOuterLoopCall = len(maskedQSpaceXX)
    breakProcess = list(chunks(range(len(maskedQSpaceXX)), num_cores*2))
    numberOfChunk = int(len(breakProcess))
    print("Total outerLoop call: ", totalOuterLoopCall)

    print("Number of Thread: " + str(num_cores))
    print("Number of Chunk: " + str(numberOfChunk))

    print("Start multiprocessing")

    processTemp = np.zeros_like(sampleCoorRealSpaceXX)
    for process in breakProcess:
        multicoreResults = Parallel(n_jobs=num_cores, backend="threading")(
            delayed(outerForLoop)(counter_i) for counter_i in process)
        tempArray = np.array(multicoreResults)
        tempArray = np.sum(tempArray, axis=0)
        processTemp = processTemp + tempArray
        printStatus(process[-1])

    print("End multiprocessing")
    matrixI = np.fft.fftshift(processTemp)
    matrixI = np.absolute(matrixI)
    print("start saving matrix")
    hkDT = datetime.now(hkTimeZone)
    timeStamp = hkDT.strftime('%Y%m%d_%H%M%S')
    np.save(timeStamp + "_alpha_ap" + "%.2f" % (alpha_ap * 1000) + ".npy", matrixI)
    print("finished saving matrix")
    print("End Main")
    printStatus(100, done=True)
    print("----------------------------------------------------")

    return matrixI


if __name__ == '__main__':
    main()
