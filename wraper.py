from datetime import timedelta
from main import *
wraperStartTime = time.time()

# from utilityFunc import *
# matrixI = main()
# plotArray(matrixI)


alpha_apList = np.linspace(1E-3, 7.5E-3, num=10)
print(alpha_apList)

loopMainCounter = 0
loopLen = len(alpha_apList)
for alp_size in alpha_apList:
    loopMainCounter += 1
    print("alpha_ap:", alp_size, loopMainCounter, "/", loopLen)
    alpha_ap = alp_size
    main()

print("Total Time: "+str(timedelta(seconds=time.time() - wraperStartTime)))