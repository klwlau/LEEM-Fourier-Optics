import numpy as np

if __name__ == '__main__':
    print("")
else:
    print("Loading Constants")



taskName = "NACSinObject"
timezone = 'Asia/Hong_Kong'
numberOfThreads = -1 # number of threads used in the simulation with -1 means using all threads, -2 means using all threads minus 1.

simulatingSpaceSize = 25 * 1e-9  # total simulating space in nm
# simulatingSpaceTotalStep = 501  # total simulating steps

# ##################### AC Constants######################
# U_a = 10.01e3  # eV  Accelerating Voltage
# U_o = 10  # eV  Electron Voltage
# C_c = 0  # m   Chromatic Aberration Coefficient
# C_3c = -67.4
# C_cc = 27.9
# C_3 = 0  # m   Spherical Aberration Coefficient
# C_5 = 92.8
#
# alpha_ap = 7.37E-3  # rad Acceptance Aangle of the Contrast Aperture
# alpha_ill = 0.1E-3  # rad Illumination Divergence Angle
#
# delta_E = 0.25  # eV  Energy Spread
# M_L = 0.653  # Lateral Magnification
# ##################### AC Constants######################


##################### NAC Constants######################
U_a = 15.01e3  # eV  Accelerating Voltage
U_o = 10  # eV  Electron Voltage
C_c = -0.075  # m   Chromatic Aberration Coefficient
C_3c = -59.37
C_cc = 23.09
C_3 = 0.345  # m   Spherical Aberration Coefficient
C_5 = 39.4

alpha_ap = 2.34E-3  # rad Acceptance Aangle of the Contrast Aperture
alpha_ill = 0.1E-3  # rad Illumination Divergence Angle

delta_E = 0.25  # eV  Energy Spread
M_L = 0.653  # Lateral Magnification
##################### NAC Constants######################


delta_z = np.linspace(-200E-6,200E-6,101)

###################################################################
delta_fc = C_c*(delta_E/U_a)
delta_f3c = C_3c*(delta_E/U_a)
delta_fcc = C_cc*(delta_E/U_a)**2

lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)
q_max = alpha_ap/lamda
q_ill = alpha_ill/lamda