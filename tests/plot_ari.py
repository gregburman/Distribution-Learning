import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'size':14})

# rc('text', usetex=True)

with open("ari/24a.txt", "rb") as fp:   # Unpickling
	s24a = pickle.load(fp)

print "mean: ", np.mean(s24a)
print "std: ", np.std(s24a)
print"median: ", np.median(s24a)
print"median elem: ", np.argsort(s24a)[len(s24a)//2]


with open("ari/24b.txt", "rb") as fp:   # Unpickling
	s24b = pickle.load(fp)

print "mean: ", np.mean(s24b)
print "std: ", np.std(s24b)
print"median elem: ", np.argsort(s24b)[len(s24b)//2]

with open("ari/24c.txt", "rb") as fp:   # Unpickling
	s24c = pickle.load(fp)

print "mean: ", np.mean(s24c)
print "std: ", np.std(s24c)
print"median elem: ", np.argsort(s24c)[len(s24c)//2]

aris = [s24a, s24b, s24c]
fig1, ax1 = plt.subplots()
# ax1.set_title('Prediction Accuracy of Baseline Model Variants')

my_xticks = ['MSE','SCE','MALIS']

# plt.xlabel("Loss Function")
plt.ylabel("ARI")
ax1.boxplot(aris, 1, '')
plt.xticks([1, 2, 3], ['MSE','SCE','MALIS'])

plt.show()