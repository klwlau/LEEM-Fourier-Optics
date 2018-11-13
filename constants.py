import numpy as np

if __name__ == '__main__':
    print("")
else:
    print("Loading Constants.py")

resultName = "NAC-_-PhaseObject"


sampleSpaceTotalStep = 501  # sample size
sampleSpaceSize = 50 * 1e-9  # nm #25 #
objectSpaceSize = 10 * 1e-9  # nm #5



U_a = 20e3  # eV  Accelerating Voltage
U_o = 10  # eV  Electron Voltage
delta_E = 0.75  # eV  Energy Spread
C_3 = 0.345  # m   Spherical Aberration Coefficient
C_5 = 0
C_cc = 0
C_3c = 0
C_c = -0.106  # m   Chromatic Aberration Coefficient
alpha_ap = 1.4E-3  # rad Acceptance Aangle of the Contrast Aperture
alpha_ill = 0.11E-3  # rad Illumination Divergence Angle

M_L = 0.653  # Lateral Magnification
defocus = 0  # mA
delta_zo = -1 * defocus * 5.23 * 10 ** -6  # m
delta_z = 0e-6  # delta_zo * 3.2



lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)




