import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d

from utils import load_data
from plotgrid import plot_grid

def extract_decision(states, usable_ace):
   decision = np.zeros(shape=(len(states), len(states[0]), 1), dtype=np.int16)
   q_value = np.zeros(shape=(len(states), len(states[0]), 1))
      
   for i, d in enumerate(states):
      for j, p in enumerate(d):
         decision[i, j] = np.argmax(p[usable_ace])
         q_value[i, j] = max(p[usable_ace])

   return (decision, q_value)

prefix = sys.argv[1]
iterations = int(sys.argv[2])

q_values = load_data(prefix=prefix, iterations=iterations)
#q_values = load_data(prefix='montecarlo', iterations=2000000)

dealer = np.arange(0, 11)
player = np.arange(0, 22)

_dealer = []
_player = []
nua_decision = []
nua_value = []
ua_decision = []
ua_value = []
iua_decision = []
iua_value = []

for d in dealer:
   for p in player:
      _dealer.append(d)
      _player.append(p)
      nua_decision.append(np.argmax(q_values[d, p, 0]))
      nua_value.append(max(q_values[d, p, 0]))
      ua_decision.append(np.argmax(q_values[d, p, 1]))
      ua_value.append(max(q_values[d, p, 1]))
      iua_decision.append(np.argmax(q_values[d, p, 2]))
      iua_value.append(max(q_values[d, p, 2]))


xi = np.linspace(0, len(dealer), len(dealer) * 2)
yi = np.linspace(0, len(player), len(player) * 2)

zi_nua_value = griddata((_dealer, _player), nua_value, (xi[None, :], yi[:, None]), method='cubic')
zi_ua_value = griddata((_dealer, _player), ua_value, (xi[None, :], yi[:, None]), method='cubic')
zi_iua_value = griddata((_dealer, _player), iua_value, (xi[None, :], yi[:, None]), method='cubic')

X, Y = np.meshgrid(xi, yi)

fig = plt.figure()

ax = fig.add_subplot(321, projection='3d')
ax.set_xlabel('X = dealer')
ax.set_ylabel('Y = player')
ax.set_zlabel('Z = nua value')
ax.set_title('VALUE - No Usable Ace')
ax.plot_wireframe(X, Y, zi_nua_value, rstride=1, cstride=1, color='blue')

ay = fig.add_subplot(322, projection='3d')
ay.set_xlabel('X = dealer')
ay.set_ylabel('Y = player')
ay.set_zlabel('Z = nua decision')
ay.set_title('DECISION - No Usable Ace')
ay.scatter(_dealer, _player, nua_decision, color='blue')

bx = fig.add_subplot(423, projection='3d')
bx.set_xlabel('X = dealer')
bx.set_ylabel('Y = player')
bx.set_zlabel('Z = ua value')
bx.set_title('VALUE - Usable Ace')
bx.plot_wireframe(X, Y, zi_ua_value, rstride=1, cstride=1, color='red')

by = fig.add_subplot(424, projection='3d')
by.set_xlabel('X = dealer')
by.set_ylabel('Y = player')
by.set_zlabel('Z = ua decision')
by.set_title('DECISION - Usable Ace')
by.scatter(_dealer, _player, ua_decision, color='red')

cx = fig.add_subplot(425, projection='3d')
cx.set_xlabel('X = dealer')
cx.set_ylabel('Y = player')
cx.set_zlabel('Z = iua value')
cx.set_title('VALUE - Idle Usable Ace')
cx.plot_wireframe(X, Y, zi_iua_value, rstride=1, cstride=1, color='green')

cy = fig.add_subplot(426, projection='3d')
cy.set_xlabel('X = dealer')
cy.set_ylabel('Y = player')
cy.set_zlabel('Z = iua decision')
cy.set_title('DECISION - Idle Usable Ace')
cy.scatter(_dealer, _player, iua_decision, color='green')

plt.suptitle('PREFIX = %s' %prefix)

plt.show()
