from utilityFunc import *
from setupObject import *
from transmittionCrossCoefficientMatrix import *

print("Start Main")


test1 = np.arange(1,10)

xx,yy = np.meshgrid(test1,test1)


print(T_o.shape)

print("End Main")