from transmittionCrossCoefficientMatrix import *

qq_i = maskedQSpaceXX + maskedQSpaceYY*1j
qq_j = maskedQSpaceXX + maskedQSpaceYY*1j

EXP = np.exp(1j*2*np.pi*((qq_i - qq_j[:,np.newaxis]).real * maskedQSpaceXX
                         +(qq_i - qq_j[:,np.newaxis]).imag * maskedQSpaceYY ))

aaa = maskedWaveObjectFT * np.conj(maskedWaveObjectFT[:,np.newaxis])
bbb = aaa.ravel()


def calI(element):
    return element * T * EXP



