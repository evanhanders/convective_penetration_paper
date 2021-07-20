import numpy as np
import matplotlib.pyplot as plt

prof = np.loadtxt('LOGS/profile10.data',skiprows=6)

i_logR = 2
i_m = 1
i_gr = 14
i_ga = 18

ga = prof[:,i_gr] / prof[:,i_ga]
r = 10**prof[:,i_logR]
print(ga)


plt.plot(r, prof[:,i_gr], label='r')
plt.plot(r, ga, c='k', label='a')
plt.legend()
#plt.xlim([0.9,1])
plt.ylim([0,2])
plt.show()