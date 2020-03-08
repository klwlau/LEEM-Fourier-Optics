from datetime import datetime
from joblib import Parallel, delayed
from FO1Dconstants import *
import multiprocessing

startTimeStamp = datetime.now().strftime('%Y%m%d_%H%M%S')

object_wavelength = 900e-9
n_sample = 1 + 2 ** 10


def create_simulated_space():
    period = 800
    simulated_space = np.linspace(-period, period, n_sample)
    return simulated_space


def create_object_fucntion():
    simulated_space = create_simulated_space()
    # #####################Step Object#####################
    # K = 1
    # h = np.zeros_like(l)
    # for counter, element in enumerate(simulated_space):
    #     if element < 0:
    #         h[counter] = 1
    # h = K * h
    # simulated_space *= 1e-9
    # phase_shift = K * h * np.pi
    # amp = 1
    # #####################Step Object#####################

    #####################Sin Object#####################
    period = 800
    kval = 30
    K = kval * np.pi
    h = K * np.pi * np.sin(2 * np.pi / period * simulated_space)
    simulated_space *= 1e-9
    phase_shift = h
    amp = 1
    #####################Sin Object#####################

    return amp * np.exp(1j * phase_shift)


def main():
    def FO1D(z, zCounter):
        R_o = np.exp(1j * 2 * np.pi * (
                C_3 * lamda ** 3 * (Q ** 4 - QQ ** 4) / 4 + C_5 * lamda ** 5 * (Q ** 6 - QQ ** 6) / 6 - z * lamda * (
                Q ** 2 - QQ ** 2) / 2))
        E_s = np.exp(-np.pi ** 2 * q_ill ** 2 * (
                C_3 * lamda ** 3 * (Q ** 3 - QQ ** 3) + C_5 * lamda ** 5 * (Q ** 5 - QQ ** 5) - z * lamda * (
                Q - QQ)) ** 2 / (4 * np.log(2)))

        AR = np.multiply(np.multiply(np.multiply(A, R_o), E_s), E_ct)

        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                matrixI[:, zCounter] = matrixI[:, zCounter] + 2 * (
                        AR[j][i] * np.exp(1j * 2 * np.pi * (Q[j][i] - QQ[j][i]) * simulated_space)).real

        matrixI[:, zCounter] = matrixI[:, zCounter] + np.trace(AR) * np.ones_like(simulated_space)

        return matrixI

    simulated_space = create_simulated_space()
    wave_obj = create_object_fucntion()

    objectFileName = "FO1DObjectWave_" + taskName + "_" + startTimeStamp + ".npy"

    F_wave_obj = np.fft.fftshift(np.fft.fft(wave_obj, n_sample) * (1 / n_sample))

    n_max = np.floor(q_max / (1 / object_wavelength))
    q = 1 / (simulated_space[1] - simulated_space[0]) * np.arange(0, n_sample, 1) / (n_sample)
    q = q - (np.max(q) - np.min(q)) / 2

    a = np.sum(np.abs(q) <= q_max)

    if len(q) > a:
        q = q[int(np.ceil(n_sample / 2 + 1 - (a - 1) / 2)):int(np.floor(n_sample / 2 + 1 + (a + 1) / 2))]
        F_wave_obj = F_wave_obj[
                     int(np.ceil(n_sample / 2 + 1 - (a - 1) / 2)):int(np.floor(n_sample / 2 + 1 + (a + 1) / 2))]

    Q, QQ = np.meshgrid(q, q)
    F_wave_obj_q, F_wave_obj_qq = np.meshgrid(F_wave_obj, np.conj(F_wave_obj))

    A = np.multiply(F_wave_obj_q, F_wave_obj_qq)
    E_cc = (1 - 1j * np.pi * delta_fcc * lamda * (Q ** 2 - QQ ** 2) / (4 * np.log(2))) ** (-0.5)
    E_ct = E_cc * np.exp(-np.pi ** 2 * (delta_fc * lamda * (Q ** 2 - QQ ** 2) + 1 / 2 * delta_f3c * lamda ** 3 * (
            Q ** 4 - QQ ** 4)) ** 2 * E_cc ** 2 / (16 * np.log(2)))

    matrixI = np.zeros((len(simulated_space), len(delta_z_series)), dtype=complex)

    print("Task:", taskName)
    print("Total Task:", len(delta_z_series))
    print("Total Parallel Steps:", np.ceil(len(delta_z_series) / (multiprocessing.cpu_count() + numberOfThreads + 1)))

    with Parallel(n_jobs=numberOfThreads, verbose=50, max_nbytes="50M") as parallel:
        parallelReult = parallel(delayed(FO1D)(z, zCounter) for zCounter, z in enumerate(delta_z_series))

    for mat in parallelReult:
        matrixI += mat

    matrixI = np.abs(matrixI)

    resultFileName = "FO1DResult_" + taskName + "_" + startTimeStamp + ".npy"

    print("Saving result to:", resultFileName)

    np.save(resultFileName, matrixI)
    print("Done")


if __name__ == '__main__':
    main()
