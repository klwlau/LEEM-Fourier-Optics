import numpy as np

if __name__ == '__main__':
    print("")
else:
    print("Loading Constants.py")

resultFileName = "resultFileName"

sampleSpaceTotalStep = 501  # sample total step size
sampleSpaceSize = 25 * 1e-9  # nm #25 #
# objectSpaceSize = 5 * 1e-9  # nm #5

U_a = 15.01e3  # eV  Accelerating Voltage
U_o = 10  # eV  Electron Voltage
delta_E = 0.25  # eV  Energy Spread
C_3 = 0  # m   Spherical Aberration Coefficient
C_5 = 92.8
C_c = 0  # m   Chromatic Aberration Coefficient
C_3c = -67.4
C_cc = 27.9
alpha_ill = 0.1E-3  # rad Illumination Divergence Angle
alpha_ap = 1.37E-3  # rad Acceptance Aangle of the Contrast Aperture
# alpha_ap = 7.37E-3  # rad Acceptance Aangle of the Contrast Aperture

M_L = 0.653  # Lateral Magnification
defocus = 0  # mA
delta_zo = -1 * defocus * 5.23 * 10 ** -6  # m
delta_z = 0e-6  # delta_zo * 3.2

###################################################################

lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)