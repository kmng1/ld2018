import sys
import numpy as np
from matplotlib import pyplot as plt

import gym

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

env = gym.make('CartPole-v0')

gamma = 1.0
alpha = 0.1
epsilon = 0.1

W = np.random.uniform(size=(2, 4))
approx = lambda state, W: np.dot(state, W.T)

def policy(state, W, epsilon):
   #act greedily
   if np.random.uniform() > epsilon:
      return np.argmax(approx(state, W))
   return np.random.randint(0, 2)

def update(state, action, reward, new_state, new_action, W, alpha):
   error = 0
   if new_action > -1:
      error = reward - (gamma * approx(new_state, W[new_action]))
   else:
      error = reward - approx(state, W[action])

   W[action] += (alpha * error * state)

total_reward = []

print('Episodes: ', EPISODES)

for e in range(EPISODES):

   terminate = False

   state = env.reset()

   action = policy(state, W, epsilon)

   trajectory = []
   reward_per_episode = 0

   while not terminate:

      env.render()

      new_state, reward, terminate, _ = env.step(action)

      if not terminate:
         new_action = policy(new_state, W, epsilon)
      else:
         new_action = -1

      reward_per_episode += 1

      update(state, action, reward, new_state, new_action, W, alpha)

      state = new_state
      action = new_action

   total_reward.append(reward_per_episode)

   print('Episode: %d, steps: %d' %(e, reward_per_episode))

   if e > 100:
      epsilon /= 2

env.close()

print('Average returns over %d episodes: %0.3f' %(EPISODES, sum(total_reward) / EPISODES))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Episodes')
ax.set_ylabel('Steps')

ax.plot(np.arange(1, len(total_reward) + 1), total_reward)

plt.show()

