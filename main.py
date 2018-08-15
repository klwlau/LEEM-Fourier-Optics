print("Main Started, Loading Libraries")
import time
from datetime import datetime
import pytz
from joblib import Parallel, delayed
import multiprocessing
from constants import *
import numexpr as ne
from numba import jit

if __name__ == '__main__':
    from utilityFunc import *

fmt = '%H:%M:%S'  # %d/%m
hkTimeZone = pytz.timezone('Asia/Hong_Kong')


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(mainPass):
    start_time = time.time()

    def printStatus(counter, done=False, loopMode=False):
        if counter != 0:
            elapsedTime = ((time.time() - start_time) / 60)
            progress = (counter / len(loopList)) * 100
            totalTime = elapsedTime / (progress / 100)
            timeLeft = totalTime - elapsedTime
            hkDT = datetime.now(hkTimeZone)
            currentHKTime = hkDT.strftime(fmt)
            if done:
                print("-Total Time: %.2f Minutes -" % elapsedTime)
            else:
                if loopMode:
                    print("Loop:", loopMainCounter, "/", loopLen,
                          "-ID:" + str(counter) + "--Elapsed Time: %.2f / %.2f min -" % (elapsedTime, totalTime)
                          + "Time Left: %.2f  min -" % timeLeft + "OuterLoop Time:%.1f s" % (
                                  elapsedTime * 60 / (counter + 1)) + "%.2f" % progress + "%--HKT:" + currentHKTime)
                else:
                    print("-ID:" + str(counter) + "--Elapsed Time: %.2f / %.2f min -" % (elapsedTime, totalTime)
                          + "Time Left: %.2f  min -" % timeLeft + "OuterLoop Time: %.1f s--" % (
                                  elapsedTime * 60 / (counter + 1)) + "%.2f" % progress + "%--HKT:" + currentHKTime)

    def createSimulatedObject():
        amp = 1

        def simulatedObjectSpaceProfile(x, y):
            value = amp * np.sin(0.00001 * (x / 3 + 100) * (y / 3 + 100)) + amp
            return value

        simulatedObject = np.zeros_like(simulatedSpace)
        for x in range(len(simulatedObject)):
            for y in range(len(simulatedObject)):
                simulatedObject[x][y] = simulatedObjectSpaceProfile(x, y)

        # simulatedObject = np.ones_like(simulatedSpace)
        return simulatedObject

    ######set up Square Object#######
    K = 10 * np.pi
    alpha_ap = mainPass
    q_max = alpha_ap / lamda
    q_ill = alpha_ill / lamda

    sampleSpaceTotalStep = 501  # sample size
    sampleSpaceSize = 25 * 1e-9  # nm #25
    objectSpaceSize = 5 * 1e-9  # nm #5

    objectStep = int((objectSpaceSize / sampleSpaceSize * sampleSpaceTotalStep) / 2)
    sampleCoorRealSpaceXX, sampleCoorRealSpaceYY = np.mgrid[-sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j,
                                                   -sampleSpaceSize:sampleSpaceSize:sampleSpaceTotalStep * 1j]

    sampleStepSize = sampleCoorRealSpaceXX[1][0] - sampleCoorRealSpaceXX[0][0]
    simulatedSpace = np.zeros(sampleCoorRealSpaceXX.shape)
    sampleCenterX, sampleCenterY = int(sampleSpaceTotalStep / 2 + 1), int(sampleSpaceTotalStep / 2 + 1)
    simulatedObjectMask = np.copy(simulatedSpace)
    simulatedObjectMask[sampleCenterX - objectStep:sampleCenterX + objectStep,
    sampleCenterY - objectStep + 30:sampleCenterY + objectStep + 30] = 1

    # simulatedObject = np.multiply(createSimulatedObject(), simulatedObjectMask)
    simulatedObject = createSimulatedObject()

    objectPhaseShift = K * simulatedObject

    np.save("objectPhaseShift.npy", objectPhaseShift)

    if __name__ == '__main__':
        plotArray(objectPhaseShift)

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

    ##############cal Matrix I##########

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

    @jit(nopython=True,cache=True) #, parallel=True
    def outerForLoop(counter_i):


        # global returnMatrix
        returnMatrix = np.zeros_like(sampleCoorRealSpaceXX,dtype=np.complex128)
        qq_i = maskedQSpaceXX[counter_i] + 1j * maskedQSpaceYY[counter_i]
        abs_qq_i = np.absolute(qq_i)

        abs_qq_i_2 = abs_qq_i ** 2
        abs_qq_i_4 = abs_qq_i_2 ** 2
        abs_qq_i_6 = abs_qq_i_2 ** 3

        for counter_j in range(len(maskedQSpaceYY)):

            if counter_i >= counter_j:

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
                E_ct_exponent = EctConstant0 * (EctConstant1 * (abs_qq_i_2 - abs_qq_j_2)
                                                + EctConstant2 * (abs_qq_i_4 - abs_qq_j_4)) ** 2
                E_ct = E_cc * np.exp(E_ct_exponent * E_cc ** 2)

                EXP_exponent = 2j * np.pi * (
                        (qq_i - qq_j).real * sampleCoorRealSpaceXX + (qq_i - qq_j).imag * sampleCoorRealSpaceYY)

                # EXP = ne.evaluate("exp(EXP_exponent)")
                EXP = np.exp(EXP_exponent)

                returnMatrix = returnMatrix + R_o * E_s * E_ct * maskedWaveObjectFT[counter_i] * np.conj(maskedWaveObjectFT[counter_j]) * EXP
                if counter_i > counter_j:
                    # EXP_exponent_sym = 2j * np.pi * (
                    #             (qq_j - qq_i).real * sampleCoorRealSpaceXX + (qq_j - qq_i).imag * sampleCoorRealSpaceYY)
                    #
                    # EXP_sym = ne.evaluate("exp(EXP_exponent_sym)")

                    R_o_sym = 1 / R_o
                    E_s_sym = E_s
                    E_cc_sym = np.sqrt(1 - EccConstant0 * (abs_qq_j_2 - abs_qq_i_2))
                    E_ct_sym = E_cc_sym * np.exp(E_ct_exponent * E_cc_sym ** 2)
                    # EXP_sym = ne.evaluate("EXP.real-1j*EXP.imag")
                    EXP_sym = EXP.real-1j*EXP.imag
                    # EXP_sym = ne.evaluate("conj(EXP)")

                    returnMatrix = returnMatrix + R_o_sym * E_s_sym * E_ct_sym * maskedWaveObjectFT[
                        counter_j] * np.conj(
                        maskedWaveObjectFT[counter_i]) * EXP_sym
            else:
                break

        return returnMatrix

    def ijSymmetry(counter_i):

        if counter_i == int(totalOuterLoopCall / 2):
            returnMatrix = outerForLoop(counter_i)
            # print(counter_i)
        else:
            returnMatrix1 = outerForLoop(counter_i)
            returnMatrix2 = outerForLoop(totalOuterLoopCall - counter_i - 1)
            returnMatrix = ne.evaluate("returnMatrix1+returnMatrix2")

        return returnMatrix

    num_cores = multiprocessing.cpu_count()
    totalOuterLoopCall = len(maskedQSpaceXX)
    loopList = list(range(len(maskedQSpaceXX)))[:int(totalOuterLoopCall / 2) + 1]
    # loopList = list(range(len(maskedQSpaceXX)))
    breakProcess = list(chunks(loopList, num_cores * 1))
    numberOfChunk = int(len(breakProcess))
    print("Total Process: ", len(loopList))

    print("Number of Thread: " + str(num_cores))
    print("Number of Chunk: " + str(numberOfChunk))

    print("Start multiprocessing")

    processTemp = np.zeros_like(sampleCoorRealSpaceXX)

    with Parallel(n_jobs=num_cores, verbose=50) as parallel:  # ,backend="threading"
        for process in breakProcess:
            # multicoreResults = parallel(delayed(outerForLoop)(counter_i) for counter_i in process)

            multicoreResults = parallel(delayed(ijSymmetry)(counter_i) for counter_i in process)
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
    matrixI = matrixI.T
    np.save(timeStamp + "_alpha_ap" + "%.2f" % (alpha_ap * 1000) + ".npy", matrixI)
    np.save("result.npy", matrixI)
    print("finished saving matrix")
    print("End Main")
    printStatus(100, done=True)
    print("----------------------------------------------------")

    return matrixI


if __name__ == '__main__':
    main(0.5E-3)
