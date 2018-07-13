import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d

from utils import load_data
from plotgrid import plot_grid

prefix = sys.argv[1]
iterations = int(sys.argv[2])

q_values = load_data(prefix=prefix, iterations=iterations)

dealer = np.arange(0, 11)
player = np.arange(0, 22)

_player = []
_dealer = []
nua_decision = []
ua_decision = []
iua_decision = []

for d in dealer:
   for p in player:
      _dealer.append(d)
      _player.append(p)
      nua_decision.append(np.argmax(q_values[d, p, 0]))
      ua_decision.append(np.argmax(q_values[d, p, 1]))
      iua_decision.append(np.argmax(q_values[d, p, 2]))

nua_fig = plt.figure(100)
nua_plot = nua_fig.add_subplot(111, projection='3d')
nua_plot.set_title('No USABLE ACE')
nua_plot.set_xlabel('Dealer - initial card')
nua_plot.set_ylabel('Player - hand')
nua_plot.set_zlabel('Decision')
nua_plot.scatter(_dealer, _player, nua_decision, color='blue')

ua_fig = plt.figure(200)
ua_plot = ua_fig.add_subplot(111, projection='3d')
ua_plot.set_title('USABLE ACE')
ua_plot.set_xlabel('Dealer - initial card')
ua_plot.set_ylabel('Player - hand')
ua_plot.set_zlabel('Decision')
ua_plot.scatter(_dealer, _player, ua_decision, color='orange')

iua_fig = plt.figure(300)
iua_plot = iua_fig.add_subplot(111, projection='3d')
iua_plot.set_title('Idle USABLE ACE')
iua_plot.set_xlabel('Dealer - initial card')
iua_plot.set_ylabel('Player - hand')
iua_plot.set_zlabel('Decision')
iua_plot.scatter(_dealer, _player, iua_decision, color='magenta')


plt.show()
