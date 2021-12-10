import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import matplotlib.pyplot as plt
import gym
import scipy.signal
import time

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self._size = size
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0
        
    def buffer_reset(self):
        self.observation_buffer = np.zeros(
            (self._size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(self._size, dtype=np.int32)
        self.advantage_buffer = np.zeros(self._size, dtype=np.float32)
        self.reward_buffer = np.zeros(self._size, dtype=np.float32)
        self.return_buffer = np.zeros(self._size, dtype=np.float32)
        self.value_buffer = np.zeros(self._size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(self._size, dtype=np.float32)
        self.pointer, self.trajectory_start_index = 0, 0
        
    def del_Redundant_data(self):
        self.observation_buffer = self.observation_buffer[:-(self._size - self.trajectory_start_index)]
        self.action_buffer = self.action_buffer[:-(self._size - self.trajectory_start_index)]
        self.reward_buffer = self.reward_buffer[:-(self._size - self.trajectory_start_index)]
        self.return_buffer = self.return_buffer[:-(self._size - self.trajectory_start_index)]
        self.value_buffer = self.value_buffer[:-(self._size - self.trajectory_start_index)]
        self.advantage_buffer = self.advantage_buffer[:-(self._size - self.trajectory_start_index)]
        self.logprobability_buffer = self.logprobability_buffer[:-(self._size - self.trajectory_start_index)]

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


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
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))
    

# In[0]

# Hyperparameters of the PPO algorithm
# steps_per_epoch = 4000
epochs = 2000
steps = 20
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

round_time = 0
total_reward = []
all_avg_reward = []

# True if you want to render the environment
# render = True

# In[1]

# Initialize the environment and get the dimensionality of the observation space and the number of possible actions
# env = gym.make("CartPole-v0")
env = KukaGymEnv(renders=False, isDiscrete=True, actionRepeat=10)
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, 10000)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)

# actor.load_weights("save_model_PPO_cube/PPO_kukaGym_actor.h5")
# critic.load_weights("save_model_PPO_cube/PPO_kukaGym_critic.h5")

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
# observation, episode_return, episode_length = env.reset(), 0, 0

# In[2]

# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    
    observation, episode_return, episode_length = env.reset(), 0, 0

    # Iterate over the steps of each epoch
    # for t in range(steps_per_epoch):
    while True:
        # if render:
        #     env.render()

        # Get the logits, action, and take one step in the environment
        blockPos_reward = 0
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        observation_new, reward, done, blockPos_high = env.step(action[0].numpy())
        if blockPos_high > 0.2:
            # new reward
            blockPos_reward = 1 
            
        # episode_return += reward
        episode_return += blockPos_reward
        episode_length += 1
        
        if episode_return >= 20 and blockPos_high > 0.2:

            actor.save_weights("./save_model_PPO_cube/PPO_kukaGym_round{}_actor_return{}.h5".format(epoch,episode_return))
            critic.save_weights("./save_model_PPO_cube/PPO_kukaGym_round{}_critic_return{}.h5".format(epoch,episode_return))
            

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, logprobability_t)

        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal :
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1     
            # total_reward.append(episode_return)
            # avg_reward = np.mean(total_reward[-40:])
            # all_avg_reward.append(avg_reward)
            round_time = round_time + 1
            if round_time % steps == 0 :
                
                total_reward.append(episode_return)
                avg_reward = np.mean(total_reward[-40:])
                all_avg_reward.append(avg_reward)
                
                plt.plot(total_reward,color='blue')
                plt.plot(all_avg_reward,color='green')
                plt.xlabel("Episode")
                plt.ylabel("Epsiodic Reward")
                if round_time % 100 == 0:
                    plt.savefig("./save_model_PPO_cube/kukaGym_picture_PPO.png")
                    # plt.show()
                break
            
            observation= env.reset()
            

    buffer.del_Redundant_data()
    
    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()
    
    buffer.buffer_reset()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
    # actor.save_weights("save_model_PPO_cube/PPO_kukaGym_actor_Finish.h5")
    # critic.save_weights("save_model_PPO_cube/PPO_kukaGym_critic_Finish.h5")
   
    
import pandas as pd

save_reward = pd.DataFrame(total_reward)
save_reward.to_csv('./save_model_PPO_cube/total_reward.csv',index=False,header = None)
    