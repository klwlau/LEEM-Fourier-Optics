import numpy as np

if __name__ == '__main__':
    print("")
else:
    print("Loading Constants.py")

U_a = 15.01e3  # eV  Accelerating Voltage
U_o = 10  # eV  Electron Voltage
C_c = 0  # m   Chromatic Aberration Coefficient
C_3c = -67.4
C_cc = 27.4
C_3 = 0  # m   Spherical Aberration Coefficient
C_5 = 92.8
alpha_ill = 0.122E-3  # rad Illumination Divergence Angle
# alpha_ap = 4.37E-3  # rad Acceptance Aangle of the Contrast Aperture
alpha_ap = 1.37E-3  # rad Acceptance Aangle of the Contrast Aperture
# alpha_ap = 7.37E-3  # rad Acceptance Aangle of the Contrast Aperture

delta_E = 0.25  # eV  Energy Spread
M_L = 0.653  # Lateral Magnification
defocus = 0  # mA

delta_zo = -1 * defocus * 5.23 * 10 ** -6  # m
delta_z = delta_zo * 3.2
delta_fc = C_c * (delta_E / U_a)
delta_f3c = C_3c * (delta_E / U_a)
delta_fcc = C_cc * (delta_E / U_a) ** 2

lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)

q_max = alpha_ap / lamda
q_ill = alpha_ill / lamda


