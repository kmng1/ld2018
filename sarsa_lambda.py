import sys, time

import numpy as np

from tqdm import *

from blackjack import Blackjack
from blackjack_biased import Blackjack_Biased
from utils import save_data, load_data, Stats, QValue, to_matrix

from matplotlib import pyplot as plt

SHUFFLE = 1000

#eligibility trace decay rate
LAMBDA = 0.5
#discound
GAMMA = 1

N0 = 100
HITS = Blackjack.HITS #0
STICK = Blackjack.HITS #1
USE_ACE = Blackjack.USE_ACE #2
IDLE_ACE = Blackjack.IDLE_ACE #3


DEFAULT_EPISODES = 3

#Number of episodes
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EPISODES
use_bias_env = True if len(sys.argv) > 2 else False

#initialize states [dealer_top_card, player_hand, usable_ace, actions]
#actions = 0 - HITS, 1 - STICK
#0 element not used
if use_bias_env:
   print('Initializing states with %s_%s.pickle' %(sys.argv[2], sys.argv[3]))
   states = load_data(prefix=sys.argv[2], iterations=int(sys.argv[3]))
else:
   states = np.zeros((11, 22, 3, 4), dtype=float)

def policy(state, q_values, stats):
   epsilon = stats.epsilon(state)
   actions = q_values[state]
   if np.random.uniform() > epsilon:
      #greedy - exploit
      return np.argmax(actions)
   #random action - explore
   return np.random.randint(4)

print('Episodes: %d' %EPISODES)
print('Biased environment: ', use_bias_env)

stats = Stats()
q_values = QValue()

win = [ 0 ]
lose = [ 0 ]
draw = [ 0 ]

wi, lo, dr, has_ace = 0, 0, 0, 0

for ep in tqdm(range(EPISODES)):

   blackjack = Blackjack_Biased(1000) if use_bias_env else Blackjack(1000)

   #get start state
   state = blackjack.reset()
   action = policy(state, q_values, stats)

   terminate = False

   trajectory = []
   stats.reset_traces()

   while not terminate:

      stats.update_stats(state, action)
      trajectory.append((state, action))

      new_state, reward, terminate = blackjack.step(action)

      if not terminate:
         new_action = policy(new_state, q_values, stats)

      else:
         new_action = -1

         if reward == 0:
            dr += 1
         elif reward == 1:
            wi += 1
         else:
            lo += 1

         has_ace += 1 if blackjack.has_usable_ace() else 0

         if (ep % 100) == 0:
            win.append(wi)
            lose.append(lo)
            draw.append(dr)
            wi, lo, dr = 0, 0, 0

      td_target = reward + (GAMMA * q_values[(new_state, new_action)])
      td_error = td_target - q_values[(state, action)]
      alpha = stats.alpha(state, action)

      #q_values[(state, action)] += stats.alpha(state, action) * td_error

      for i in trajectory:
         st, ac = i
         q_values[(st, ac)] += alpha * td_error * stats.trace[(st, ac)]
         stats.decay_trace(st, ac, GAMMA, LAMBDA)

      state = new_state
      action = new_action

print('win: %d, lose: %d, draw: %d' %(sum(win), sum(lose), sum(draw)))
print('win: %0.1f%%, lose: %0.1f%%, draw: %0.1f%%' \
      %(sum(win)/EPISODES * 100, sum(lose)/EPISODES * 100, sum(draw)/EPISODES * 100))
print('usable ace: %0.2f%%' %(has_ace/EPISODES * 100))

q_value_matrix = to_matrix(q_values.dump_values())

fn = save_data(data=q_value_matrix \
      , prefix='sarsa-lambda' + ('-biased' if use_bias_env else ''), iterations=EPISODES)
print('Saved %s' %fn)

x = np.arange(len(win))

plt.scatter(x, win, label='WIN', color='green', s=1)
plt.scatter(x, lose, label='LOSE', color='red', s=1)
plt.scatter(x, draw, label='DRAW', color='blue', s=1)

l = u'λ'
plt.title('SARSA(%s=%0.1f)' %(u'λ', LAMBDA))

plt.legend()

plt.show()
