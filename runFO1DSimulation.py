from datetime import datetime
import pytz
from joblib import Parallel, delayed
from FO1Dconstants import *
import multiprocessing

fmt = '%H:%M:%S'  # %d/%m
timeZonePytz = pytz.timezone(timezone)
startTimeStamp = datetime.now(timeZonePytz).strftime('%Y%m%d_%H%M%S')

object_wavelength = 900e-9
n_sample = 1 + 2 ** 10
# l = np.linspace(-object_wavelength, object_wavelength, n_sample)

# KList = np.array([3, 4, 5, 6]) * 2
# n_max = np.floor(q_max / (1 / object_wavelength))
# q = (1 / object_wavelength) * np.arange(-n_max, n_max, 1)
#
# q = q.T

# kVal = 1
#
# print("Start kVal:", kVal)
#
# F_wave_obj = np.zeros(int(2 * n_max))
#
# for i in range(len(F_wave_obj)):
#     n = i - (n_max + 1)
#     if n < 0:
#         F_wave_obj[i] = (-1) ** (abs(n)) * special.jv(abs(n), kVal * np.pi)
#     else:
#         F_wave_obj[i] = special.jv(n, kVal * np.pi)

period = 800
l = np.linspace(-period, period, n_sample)

# ##Step Object
# K = 1
# h = np.zeros_like(l)
# for counter, element in enumerate(l):
#     if element < 0:
#         h[counter] = 1
# h = K * h
# l *= 1e-9
# phase_shift = K * h * np.pi
# amp = 1

# # Sin Object
K = 10*np.pi

period = 900
l = np.linspace(-period, period, n_sample)
h = K*np.pi*np.sin(2*np.pi/period*l)
l = l*1e-9
phase_shift = h
amp = 1


# Main simulation

wave_obj = amp * np.exp(1j * phase_shift)
F_wave_obj = np.fft.fftshift(np.fft.fft(wave_obj, n_sample)*(1/n_sample))
q = 1 / (l[1] - l[0]) * np.arange(0, n_sample, 1) / (n_sample - 1)
q = q - (np.max(q) - np.min(q)) / 2
q = q.T

a = np.sum(np.abs(q) <= q_max)


if len(q)>a:
    q = q[int(np.ceil(n_sample/2+1-(a-1)/2)):int(np.floor(n_sample/2+1+(a+1)/2))]
    F_wave_obj = F_wave_obj[int(np.ceil(n_sample / 2 + 1 - (a - 1) / 2)):int(np.floor(n_sample / 2 + 1 + (a + 1) / 2))]

Q, QQ = np.meshgrid(q, q)
F_wave_obj_q, F_wave_obj_qq = np.meshgrid(F_wave_obj, np.conj(F_wave_obj))

A = np.multiply(F_wave_obj_q, F_wave_obj_qq)
E_cc = (1 - 1j * np.pi * delta_fcc * lamda * (Q ** 2 - QQ ** 2) / (4 * np.log(2))) ** (-0.5)
E_ct = E_cc * np.exp(-np.pi ** 2 * (delta_fc * lamda * (Q ** 2 - QQ ** 2) + 1 / 2 * delta_f3c * lamda ** 3 * (
        Q ** 4 - QQ ** 4)) ** 2 * E_cc ** 2 / (16 * np.log(2)))

matrixI = np.zeros((len(l), len(delta_z)))


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
                    AR[j][i] * np.exp(1j * 2 * np.pi * (Q[j][i] - QQ[j][i]) * l)).real

    return matrixI


print("Task:",taskName)
print("Total Task:", len(delta_z))
print("Total Parallel Steps:", np.ceil(len(delta_z)/(multiprocessing.cpu_count()+numberOfThreads+1)))

# z = delta_z[0]
# zCounter = 0
# R_o = np.exp(1j * 2 * np.pi * (
#         C_3 * lamda ** 3 * (Q ** 4 - QQ ** 4) / 4 + C_5 * lamda ** 5 * (Q ** 6 - QQ ** 6) / 6 - z * lamda * (
#         Q ** 2 - QQ ** 2) / 2))
# E_s = np.exp(-np.pi ** 2 * q_ill ** 2 * (
#         C_3 * lamda ** 3 * (Q ** 3 - QQ ** 3) + C_5 * lamda ** 5 * (Q ** 5 - QQ ** 5) - z * lamda * (
#         Q - QQ)) ** 2 / (4 * np.log(2)))
# AR = np.multiply(np.multiply(np.multiply(A, R_o), E_s), E_ct)
#
# for i in range(len(q)):
#     for j in range(i + 1, len(q)):
#         matrixI[:, zCounter] = matrixI[:, zCounter] + 2 * (
#                 AR[j][i] * np.exp(1j * 2 * np.pi * (Q[j][i] - QQ[j][i]) * l)).real


with Parallel(n_jobs=numberOfThreads, verbose=50) as parallel:
    parallelReult = parallel(delayed(FO1D)(z, zCounter) for zCounter, z in enumerate(delta_z))


for mat in parallelReult:
    matrixI += mat

matrixI = np.abs(matrixI)


resultFileName = "FO1Dresult_" + taskName + "_" + startTimeStamp + ".npy"
print("Saving result to:",resultFileName)

np.save(resultFileName, matrixI)
