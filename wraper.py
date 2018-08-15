from datetime import timedelta
from main import *
wraperStartTime = time.time()

#########################################################################################################
# matrixI = main(alpha_ap)
# from utilityFunc import *
# plotArray(matrixI)

#########################################################################################################


defocusList = np.linspace(-7, 7, num=15)
print(defocusList)

loopMainCounter = 0
loopLen = len(defocusList)
for defocus in defocusList:
    loopMainCounter += 1
    print("defocus:", defocus, loopMainCounter, "/", loopLen)
    main(defocus)

print("Total Time: "+str(timedelta(seconds=time.time() - wraperStartTime)))