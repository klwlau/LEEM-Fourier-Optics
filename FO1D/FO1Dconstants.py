import numpy as np

if __name__ == '__main__':
    print("")
else:
    print("Loading Constants")



resultFileNote = ""
timezone = 'Asia/Hong_Kong'
numberOfThreads = -1 # number of threads used in the simulation with -1 means using all threads, -2 means using all threads minus 1.

simulatingSpaceSize = 25 * 1e-9  # total simulating space in nm
# simulatingSpaceTotalStep = 501  # total simulating steps


U_a = 15.01e3  # eV  Accelerating Voltage
U_o = 10  # eV  Electron Voltage
delta_E = 0.25  # eV  Energy Spread
C_3 = 0  # m   Spherical Aberration Coefficient
C_5 = 92.8
C_c = 0  # m   Chromatic Aberration Coefficient
C_3c = -67.4
C_cc = 27.9
alpha_ill = 0.1E-3  # rad Illumination Divergence Angle
alpha_ap = 7.37E-3  # rad Acceptance Aangle of the Contrast Aperture

M_L = 0.653  # Lateral Magnification
# defocus = 0  # mA
# delta_zo = -1 * defocus * 5.23 * 10 ** -6  # m
# delta_z = 0e-6  # delta_zo * 3.2

delta_z = np.arange(-120e-6,120e-6,2e-6)
delta_fc = C_c*(delta_E/U_a)
delta_f3c = C_3c*(delta_E/U_a)
delta_fcc = C_cc*(delta_E/U_a)**2

###################################################################

lamda = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_a)
lamda_o = 6.6262e-34 / np.sqrt(2 * 1.6022e-19 * 9.1095e-31 * U_o)
q_max = alpha_ap/lamda
q_ill = alpha_ill/lamda