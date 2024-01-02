# Load OpenAI Gym and other necessary packages
import gym
import numpy

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Training parameters for Q learning
alpha = 0.9  # Learning rate
gamma = 0.9  # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500  # per each episode
epsilon = 0.2

# Q tables for rewards
# Q_reward = -100000*numpy.ones((500,6)) # All same
# Q_reward = -100000 * numpy.random.random((500, 6))  # Random

# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE
state_size = env.observation_space.n
action_size = env.action_space.n
qtable = numpy.random.rand(state_size, action_size)

for episode in range(num_of_episodes):
    state = env.reset()[0]
    done = False

    for step in range(num_of_steps):

        if numpy.random.uniform(0, 1) > epsilon:
            action = numpy.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, tc, info = env.step(action)

        if done:
            qtable[state, action] = reward
            break
        else:
            qtable[state, action] = (1 - alpha) * qtable[
                state, action] + alpha * (reward + gamma * numpy.max(
                qtable[new_state, :]))

            state = new_state

# Testing
env.reset()
rewards = []
all_steps = []
for episode in range(10):
    state = env.reset()[0]
    tot_reward = 0
    steps = 0
    for t in range(50):
        action = numpy.argmax(qtable[state, :])
        state, reward, done, truncated, info = env.step(action)
        tot_reward += reward
        steps += 1
        print(env.render())
        # time.sleep(1)
        if done:
            rewards.append(tot_reward)
            all_steps.append(steps)
            #print("Total reward %d" % tot_reward)
            #print("Total steps %d" % steps)
            break

env.close()
print("Averages of 10 runs \n--------------------")
print("Average rewards: " + str(sum(rewards) / 10))
print("Average actions: " + str(sum(all_steps) / 10))
