import time
from datetime import datetime
import pytz
from joblib import Parallel, delayed
import multiprocessing
from FO1Dconstants import *
from scipy import special
from pytictoc import TicToc

timer = TicToc()

timer.tic()

object_wavelength = 900e-9
n_sample = 1 + 2 ** 10
l = np.linspace(-object_wavelength, object_wavelength, n_sample)

K = [3, 4, 5, 6] * 2
n_max = np.floor(q_max / (1 / object_wavelength))
q = (1 / object_wavelength) * np.arange(-n_max, n_max, 1)

q = q.T

kCounter = 1
# for kCounter, k in enumerate(K):

F_wave_obj = np.zeros(int(2 * n_max))

for i in range(len(F_wave_obj)):
    n = i - (n_max + 1)
    if n < 0:
        F_wave_obj[i] = (-1) ** (abs(n)) * special.jv(abs(n), K[kCounter] * np.pi)
    else:
        F_wave_obj[i] = special.jv(n, K[kCounter] * np.pi)

Q, QQ = np.meshgrid(q, q)
F_wave_obj_q, F_wave_obj_qq = np.meshgrid(F_wave_obj, np.conj(F_wave_obj))
A = np.multiply(F_wave_obj_q, F_wave_obj_qq)

E_cc = (1 - 1j * np.pi * delta_fcc * lamda * (Q ** 2 - QQ ** 2) / (4 * np.log(2))) ** (-0.5)
E_ct = E_cc * np.exp(-np.pi ** 2 * (delta_fc * lamda * (Q ** 2 - QQ ** 2) + 1 / 2 * delta_f3c * lamda ** 3 * (
            Q ** 4 - QQ ** 4)) ** 2 * E_cc ** 2 / (16 * np.log(2)))

I = np.zeros((len(l), len(delta_z)))

for zCounter, z in enumerate(delta_z):
    R_o = np.exp(1j * 2 * np.pi * (
                C_3 * lamda ** 3 * (Q ** 4 - QQ ** 4) / 4 + C_5 * lamda ** 5 * (Q ** 6 - QQ ** 6) / 6 - z * lamda * (
                    Q ** 2 - QQ ** 2) / 2))
    E_s = np.exp(-np.pi ** 2 * q_ill ** 2 * (
                C_3 * lamda ** 3 * (Q ** 3 - QQ ** 3) + C_5 * lamda ** 5 * (Q ** 5 - QQ ** 5) - z * lamda * (
                    Q - QQ)) ** 2 / (4 * np.log(2)))
    AR = np.multiply(np.multiply(np.multiply(A, R_o), E_s), E_ct)

    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            I[:, zCounter] = I[:, zCounter] + 2 * (AR[j][i] * np.exp(1j * 2 * np.pi * (Q[j][i] - QQ[j][i]) * l)).real

timer.toc()
