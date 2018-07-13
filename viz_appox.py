import sys, pickle
import numpy as np

from matplotlib import pyplot as plt

from utils import load_data, min_max_scaler
from plotgrid import plot_grid

#prefix = sys.argv[1]
#iterations = int(sys.argv[2])

#W = load_data(prefix=prefix, interations=iterations)
W = pickle.load(open('/opt/tmp/montecarlo-approx_500000.pickle', 'rb'))

labels = [ "STICK", "HIT", "USE_ACE", "IDLE_ACE" ]
Y = [ [], [], [], [] ]

for d in range(1, 11):
   for p in range(1, 22):
      for u in range(0, 3):
         st = np.array(min_max_scaler((d, p, u)) + (1,))
         for a in range(0, 4):
            Y[a].append(np.dot(W[a], st.T))

X = np.arange(len(Y[0]))

for i in range(len(Y)):
   plt.scatter(X, Y[i], label=labels[i], s=1)

plt.legend()
plt.show()
