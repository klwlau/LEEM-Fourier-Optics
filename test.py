from joblib import Parallel, delayed

from tqdm import tqdm

def myfun(x):
    return x**2

results = Parallel(n_jobs=4)(delayed(myfun)(i) for i in tqdm(range(100000)))
