import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("ari/24a.txt", "rb") as fp:   # Unpickling
	s24a = pickle.load(fp)

with open("ari/24c.txt", "rb") as fp:   # Unpickling
	s24b = pickle.load(fp)

with open("ari/24c.txt", "rb") as fp:   # Unpickling
	s24c = pickle.load(fp)

aris = [s24a, s24b, s24c]
fig1, ax1 = plt.subplots()
ax1.set_title('ARI Scores of Baseline Models')
ax1.boxplot(aris)

plt.show()