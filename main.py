from calMatrixI import *
from joblib import Parallel, delayed
import multiprocessing

print("Start Main")
num_cores = multiprocessing.cpu_count()

multicoreResults = Parallel(n_jobs=num_cores)(delayed(calI)(element) for element in abs_maskedWaveObjectFT)
# for result in multicoreResults:
#     result += result
#
# matrixI = np.fft.fftshift(result)

print("End Main")
