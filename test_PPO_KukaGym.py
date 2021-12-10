import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import matplotlib.pyplot as plt
import gym
import scipy.signal
import time


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
# @tf.function



# Train the value function by regression on mean-squared error

    

# In[0]

# Hyperparameters of the PPO algorithm
# steps_per_epoch = 4000
epochs = 100
hidden_sizes = (64, 64)
total_reward = []


# True if you want to render the environment
# render = True

# In[1]

# Initialize the environment and get the dimensionality of the observation space and the number of possible actions

env = KukaGymEnv(renders=True, isDiscrete=True, actionRepeat=10)
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n


# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)

actor.load_weights("test_model/PPO_kukaGym_round1950_actor_return20.h5")
critic.load_weights("test_model/PPO_kukaGym_round1950_critic_return20.h5")



# In[2]

# Iterate over the number of epochs
for epoch in range(epochs):

    observation, episode_return,  = env.reset(), 0
    
    while True:

        blockPos_reward = 0
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        observation_new, reward, done, blockPos_high = env.step(action[0].numpy())
        if blockPos_high >= 0.2:
            blockPos_reward = 1
            
        episode_return += blockPos_reward

        observation = observation_new

        terminal = done
        if terminal :
            last_value = 0 if done else critic(observation.reshape(1, -1))
            total_reward.append(episode_return)
            observation, episode_return, = env.reset(), 0
            
            break
    avg_reward = (np.mean(total_reward)*100)
    print("round{}: {:.2f}%".format(epoch+1, avg_reward))
    
            

    