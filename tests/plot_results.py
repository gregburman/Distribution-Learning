import pickle
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

with open("results/setup_Q1.txt", "rb") as fp:   # Unpickling
	q1 = pickle.load(fp)

with open("results/setup_Q2.txt", "rb") as fp:   # Unpickling
	q1 = pickle.load(fp)

with open("results/setup_Q2.txt", "rb") as fp:   # Unpickling
	q3 = pickle.load(fp)

with open("results/setup_Q2.txt", "rb") as fp:   # Unpickling
	q4 = pickle.load(fp)

# print (x)

# print "mean: ", np.mean(s24a)
# print "std: ", np.std(s24a)
# print"median: ", np.median(s24a)
# print"median elem: ", np.argsort(s24a)[len(s24a)//2]

# print "mean: ", np.mean(s24c)
# print "std: ", np.std(s24c)
# print"median elem: ", np.argsort(s24c)[len(s24c)//2]

pred_acc = [q1['ari_YS']]
pred_var = []
ged = []
fig1, ax1 = plt.subplots()
# ax1.set_title('Prediction Accuracy of Baseline Model Variants')

plt.xlabel("Loss Function")
plt.ylabel("ARI")
ax1.boxplot(pred_acc)
plt.xticks([1, 2, 3], ['Q1','Q2','Q3', 'Q4'])

plt.show()