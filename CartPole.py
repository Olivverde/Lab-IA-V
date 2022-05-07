import numpy as np 
import gym 
import time
import math 

# Carpole enviroment has been added
env = gym.make("CartPole-v1")
print(env.action_space.n)

# Initial designated variables 
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 60000
total = 0
total_reward = 0
prior_reward = 0
Observation = [30, 30, 50, 50] # Cart position | Cart Velocity | Pole Angle | Pole Velocity
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1]) # Steps for observations vars
# Epsilon values
epsilon = 1
epsilon_decay_value = 0.99995
# q-table
q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
q_table.shape

# Gets the discrete state
def get_discrete_state(state):
    discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES + 1): 
    t0 = time.time() # Added to give time to balance the pole
    discrete_state = get_discrete_state(env.reset()) 
    done = False
    episode_reward = 0 

    if episode % 2000 == 0: 
        print("Episode: " + str(episode))

    while not done: 

        if np.random.random() > epsilon:
            # Get cordinated action
            action = np.argmax(q_table[discrete_state])
        else:
            # Perform random action
            action = np.random.randint(0, env.action_space.n)
        # States new actions
        new_state, reward, done, _ = env.step(action) 
        # States reward 
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % 2000 == 0:
            env.render()
            print(q_table)
        # q-table has been updated
        if not done: 
            max_future_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state
    # Epsilon modification
    if epsilon > 0.05: 
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))

    t1 = time.time() 
    episode_total = t1 - t0 
    total = total + episode_total

    total_reward += episode_reward 
    prior_reward = episode_reward
    # Perform 1k episodes after showing avg time & reward
    if episode % 1000 == 0: 
        mean = total / 1000
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

env.close()
