# from utilityFunc import *
from calMatI import *
from joblib import Parallel, delayed
import multiprocessing

print("Start Main")
num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(calI)(element) for element in bbb)
for result in results:
    result += result
result = np.fft.fftshift(result)

print("End Main")