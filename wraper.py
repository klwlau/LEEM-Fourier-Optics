from main import *


alpha_apList = np.linspace(0.5E-3, 1E-3, num=10)
print(alpha_apList)

loopMainCounter = 0
loopLen = len(alpha_apList)
for alp_size in alpha_apList:
    loopMainCounter += 1
    print("alpha_ap:", alp_size, loopMainCounter,"/",loopLen)
    alpha_ap = alp_size
    main()
