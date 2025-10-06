import numpy as np
import pylab as plt


noRandomFile = '../ACC_test_10d_11552.txt'
RandomFile = []
for i in range(1,10):
	RandomFile.append(f'ACC_test_10d_11552_seed{i}.txt')
plotFile = 'test_10d_11552.pdf'

# Calculate the averaged accuracy of the optimal circuit
Acc0 = np.genfromtxt(noRandomFile,usecols=0,skip_header=1)
Acc1 = np.genfromtxt(noRandomFile,usecols=1,skip_header=1)
Acc0 = np.array([0.5,*Acc0])
Acc1 = np.array([0.5,*Acc1])
AccAver_best=(Acc0+Acc1)*0.5

# Calculate the averaged 'average accuracy'
AccAver_collect = []
for i in range(len(RandomFile)):
	Acc0 = np.genfromtxt(RandomFile[i],usecols=0,skip_header=1)
	Acc1 = np.genfromtxt(RandomFile[i],usecols=1,skip_header=1)
	Acc0 = np.array([0.5,*Acc0])
	Acc1 = np.array([0.5,*Acc1])
	AccAver_collect.append((Acc0+Acc1)*0.5)
AccAver_aver = np.zeros(len(AccAver_collect[0]))
for i in range(len(AccAver_collect)):
	for j in range(len(AccAver_collect[i])):
		AccAver_aver[j]+=AccAver_collect[i][j]/len(AccAver_collect)	#average the averaged accuracy of every gate
AccAver_err = np.zeros(len(AccAver_collect[0]))
for i in range(len(AccAver_collect)):
	for j in range(len(AccAver_collect[i])):
		AccAver_err[j] += (AccAver_aver[j]-AccAver_collect[i][j])**2/(len(AccAver_collect)-1)
AccAver_err = np.sqrt(AccAver_err)


gates = np.linspace(0,len(AccAver_best)-1,len(AccAver_best))

fig = plt.figure(figsize=(11.69*0.4,8.27*0.4))


plt.plot(gates,AccAver_best,'-o',label='optimized angles',color='red')
plt.errorbar(gates,AccAver_aver,AccAver_err,fmt='-d',label='random angles',color='blue')
plt.xlabel('gate')
plt.ylabel('<accuracy>')
plt.xlim(min(gates),max(gates))
plt.axhline(0.5,color='black')
plt.ylim(0.2,1)
plt.legend()
plt.tight_layout()
plt.savefig(plotFile)
plt.show()
