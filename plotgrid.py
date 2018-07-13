import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d

def plot_grid(*, dealer, player, value, decision):

   _dealer = []
   _player = []
   _value = []
   _decision = []

   for d in dealer:
      for p in player:
         _dealer.append(d)
         _player.append(p)
         _value.append(value[d, p, 0])
         _decision.append(decision[d, p, 0])

   xi = np.linspace(0, len(dealer), len(dealer) * 2)
   yi = np.linspace(0, len(player), len(player) * 2)
   zi_value = griddata((_dealer, _player), _value, (xi[None, :], yi[:, None]), method='cubic')
   zi_decision = griddata((_dealer, _player), _decision, (xi[None, :], yi[:, None]), method='cubic')

   X, Y = np.meshgrid(xi, yi)

   fig = plt.figure()

   ax = fig.add_subplot(121, projection='3d')
   ax.set_xlabel('X = dealer')
   ax.set_ylabel('Y = player')
   ax.set_zlabel('Z = value')
   ax.set_title('Value')

   ax.plot_wireframe(X, Y, zi_value, rstride=1, cstride=1, color='blue')

   ay = fig.add_subplot(122, projection='3d')
   ay.set_xlabel('X = dealer')
   ay.set_ylabel('Y = player')
   ay.set_zlabel('Z = decision')
   ay.set_title('Decision')

   ay.scatter(_dealer, _player, _decision)

   #ay.plot_wireframe(X, Y, zi_decision, rstride=1, cstride=1, color='green')

   plt.show()
