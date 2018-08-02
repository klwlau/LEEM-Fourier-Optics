from setupObject import *

# T = np.zeros(N, N)
maskedQSpaceXX = maskedQSpaceXX[251-72:251+72,251-72:251+72]
maskedQSpaceYY = maskedQSpaceYY[251-72:251+72,251-72:251+72]

Qx_i, Qx_j = np.meshgrid(maskedQSpaceXX, maskedQSpaceXX, sparse=True)
Qy_i, Qy_j = np.meshgrid(maskedQSpaceYY, maskedQSpaceYY, sparse=True)
F_i, F_j = np.meshgrid(maskedWaveObjectFT, maskedWaveObjectFT, sparse=True)

Qi = Qx_i + 1j * Qy_i
Qj = Qx_j + 1j * Qy_j

abs_Qi = (Qx_i ** 2 + Qy_i ** 2) ** 0.5
abs_Qj = (Qx_j ** 2 + Qy_j ** 2) ** 0.5

T_o = np.exp(1j * 2 * np.pi * (1 / 4 * C_3 * lamda ^ 3 * (abs_Qi ** 4 - abs_Qj ** 4)
                               + 1 / 6 * C_5 * lamda ^ 5 * (abs_Qi ** 6 - abs_Qj ** 6)
                               - 1 / 2 * delta_z * lamda * (abs_Qi ** 2 - abs_Qj ** 2)
                               ))
#
# E_s = np.exp(-np.pi ^ 2 / 4 / log(2) * q_ill ^ 2 *
# abs(C_3 * lamda ^ 3 * (Qi. * abs_Qi** 2 - Qj. * abs_Qj** 2)
#     + C_5 * lamda ^ 5 * (Qi. * abs_Qi** 4 - Qj. * abs_Qj** 4)
#     - delta_z * lamda * (Qi - Qj))** 2
# )
#
# E_cc = (1 - 1i * np.pi / 4 / log(2) *
# delta_fcc * lamda * (abs_Qi** 2 - abs_Qj** 2)
# )** -0.5
#
# E_ct = E_cc. *
# np.exp(-np.pi ^ 2 / 16 / log(2) *
# (delta_fc * lamda * (abs_Qi** 2 - abs_Qj** 2)
#  + 1 / 2 * delta_f3c * lamda ^ 3 * (abs_Qi** 4 - abs_Qj** 4))** 2
#     . * E_cc** 2)
#
# T = T_o. * E_s. * E_ct
