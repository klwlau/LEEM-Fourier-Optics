print("Main Started, Loading Libraries")
import time
from datetime import datetime
import pytz
from joblib import Parallel, delayed
import multiprocessing
from numba import jit
from constants2DFO import *

if __name__ == '__main__':
    from utilityFunc import *

fmt = '%H:%M:%S'  # %d/%m
timeZonePytz = pytz.timezone(timezone)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    start_time = time.time()
    from sample2D import create2DSimulatedObject

    def printStatus(counter, done=False):
        if counter != 0:
            elapsedTime = ((time.time() - start_time) / 60)
            progress = (counter / len(loopList)) * 100
            totalTime = elapsedTime / (progress / 100)
            timeLeft = totalTime - elapsedTime
            nowDT = datetime.now(timeZonePytz)
            currentHKTime = nowDT.strftime(fmt)
            if done:
                print("-Total Time: %.2f Minutes -" % elapsedTime)
            else:
                print("-ID:" + str(counter) + "---Elapsed Time: %.2f / %.2f min---" % (elapsedTime, totalTime)
                          + "Time Left: %.2f  min---" % timeLeft + "%.2f" % progress + "%-- Time: " + currentHKTime)



    ######set up Object#######
    K = 1 * np.pi
    q_max = alpha_ap / lamda
    q_ill = alpha_ill / lamda

    # defocus = mainPass

    # objectMaskStep = int((objectSpaceSize / sampleSpaceSize * sampleSpaceTotalStep) / 2)
    sampleCoorRealSpaceXX, sampleCoorRealSpaceYY = np.mgrid[-simulatingSpaceSize:simulatingSpaceSize:simulatingSpaceTotalStep * 1j,
                                                   -simulatingSpaceSize:simulatingSpaceSize:simulatingSpaceTotalStep * 1j]

    sampleStepSize = sampleCoorRealSpaceXX[1][0] - sampleCoorRealSpaceXX[0][0]
    simulatedSpace = np.zeros(sampleCoorRealSpaceXX.shape)
    # sampleCenterX, sampleCenterY = int(sampleSpaceTotalStep / 2 + 1), int(sampleSpaceTotalStep / 2 + 1)
    # simulatedObjectMask = np.copy(simulatedSpace)
    # simulatedObjectMask[sampleCenterX - objectMaskStep:sampleCenterX + objectMaskStep,
    # sampleCenterY - objectMaskStep + 30:sampleCenterY + objectMaskStep + 30] = 1

    # simulatedObject = np.multiply(create2DSimulatedObject(simulatedSpace), simulatedObjectMask)

    objectPhaseShift = K * create2DSimulatedObject(simulatedSpace)

    np.save("simObject.npy", objectPhaseShift)

    # apply wave function and apply FFT
    amp = 1
    Object = amp * np.exp(1j * objectPhaseShift)

    ObjectFT = np.fft.fftshift(np.fft.fft2(Object) / simulatingSpaceTotalStep ** 2)

    # setup qSpace
    qSpaceCoor = 1 / sampleStepSize / (simulatingSpaceTotalStep - 1) * np.arange(simulatingSpaceTotalStep)
    qSpaceCoor = qSpaceCoor - (np.amax(qSpaceCoor) - np.amin(qSpaceCoor)) / 2  # adjust qSpaceCoor center

    qSpaceXX, qSpaceYY = np.meshgrid(qSpaceCoor, qSpaceCoor)

    # setup aperture function
    apertureMask = qSpaceXX ** 2 + qSpaceYY ** 2 <= q_max ** 2
    aperture = np.zeros_like(qSpaceYY)
    aperture[apertureMask] = 1

    # apply aperture function
    maskedWaveObjectFT = ObjectFT[aperture == 1]

    maskedQSpaceXX = qSpaceXX[aperture == 1]
    maskedQSpaceYY = qSpaceYY[aperture == 1]


    print("making transmittion CrossCoefficientMatrix")

    ##############cal Matrix I##########

    delta_fc = C_c * (delta_E / U_a)
    delta_f3c = C_3c * (delta_E / U_a)
    delta_fcc = C_cc * (delta_E / U_a) ** 2

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

    @jit(nopython=True, cache=True)
    def outerForLoop(counter_i):
        returnMatrix = np.zeros_like(sampleCoorRealSpaceXX, dtype=np.complex128)

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

                EXP = np.exp(EXP_exponent)
                returnMatrix = returnMatrix + R_o * E_s * E_ct * maskedWaveObjectFT[counter_i] * np.conj(
                    maskedWaveObjectFT[counter_j]) * EXP
                if counter_i > counter_j:

                    R_o_sym = 1 / R_o
                    E_s_sym = E_s
                    E_cc_sym = np.sqrt(1 - EccConstant0 * (abs_qq_j_2 - abs_qq_i_2))
                    E_ct_sym = E_cc_sym * np.exp(E_ct_exponent * E_cc_sym ** 2)

                    EXP_sym = EXP.real - 1j * EXP.imag


                    returnMatrix = returnMatrix + R_o_sym * E_s_sym * E_ct_sym * maskedWaveObjectFT[
                        counter_j] * np.conj(
                        maskedWaveObjectFT[counter_i]) * EXP_sym
            else:
                break

        return returnMatrix

    def ijSymmetry(counter_i):

        if counter_i == int(totalOuterLoopCall / 2):
            returnMatrix = outerForLoop(counter_i)
        else:
            returnMatrix1 = outerForLoop(counter_i)
            returnMatrix2 = outerForLoop(totalOuterLoopCall - counter_i - 1)
            returnMatrix = returnMatrix1+returnMatrix2

        return returnMatrix

    num_cores = multiprocessing.cpu_count()
    totalOuterLoopCall = len(maskedQSpaceXX)
    loopList = list(range(totalOuterLoopCall))[:int(totalOuterLoopCall / 2) + 1]

    breakProcess = list(chunks(loopList, num_cores * 2))
    numberOfChunk = int(len(breakProcess))
    print("Total Process: ", len(loopList))

    print("Number of Thread: " + str(num_cores))
    print("Number of Chunk: " + str(numberOfChunk))

    print("Start multiprocessing")

    processTemp = np.zeros_like(sampleCoorRealSpaceXX)

    with Parallel(n_jobs=-1, verbose=50) as parallel:
        for process in breakProcess:

            multicoreResults = parallel(delayed(ijSymmetry)(counter_i) for counter_i in process)
            tempArray = np.array(multicoreResults)
            tempArray = np.sum(tempArray, axis=0)
            processTemp = processTemp + tempArray
            printStatus(process[-1])

    print("End multiprocessing")
    matrixI = np.fft.fftshift(processTemp)
    matrixI = np.absolute(matrixI)
    print("start saving matrix")
    nowDT = datetime.now(timeZonePytz)
    timeStamp = nowDT.strftime('%Y%m%d_%H%M%S')
    matrixI = matrixI.T
    np.save(resultFileName + ".npy", matrixI)
    np.save("result_"+timeStamp+".npy", matrixI)
    print("finished saving matrix")
    print("End Main")
    printStatus(100, done=True)
    print("----------------------------------------------------")

    return matrixI


if __name__ == '__main__':
    main()
