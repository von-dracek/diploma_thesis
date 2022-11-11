import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import random
import logging

from log import configure
from reinforcement_environment import TreeBuildingEnv

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
configure()
env = TreeBuildingEnv()
np.random.seed(RANDOM_SEED)
logger = logging.getLogger("LOGGER")
logger.info("Action Space: {}".format(env.action_space))
logger.info("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 10**6
test_episodes = 100

#reference implementation: https://github.com/mswang12/minDQN/blob/main/minDQN.py
#https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.zeros()
    model = keras.Sequential()
    model.add(keras.layers.Dense(48, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(24, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='linear', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model



def train(replay_memory, model, target_model):
    learning_rate = 0.5  # Learning rate
    discount_factor = 0.99

    MIN_REPLAY_SIZE = 64*4
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    logger.debug("Training the model")

    batch_size = 64*4
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0].flatten() for transition in mini_batch])
    current_qs_list = model.predict(current_states, verbose=0)
    new_current_states = np.array([transition[3].flatten() for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states, verbose=0)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        act_index = action - 2
        current_qs = current_qs_list[index]
        current_qs[act_index] = (1 - learning_rate) * current_qs[act_index] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    x_arr = np.array(X)
    x_arr = x_arr.reshape(batch_size, -1)
    y_arr = np.array(Y)
    model.fit(x_arr, y_arr, batch_size=batch_size, verbose=0, shuffle=True)


def _get_action_from_predicted_arr(predictions: np.ndarray):
    return np.argmax(predictions) + 2

def main():
    epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1  # You can't explore more than 100% of the time
    min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
    decay = 0.0005

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    model = agent((np.prod(env.S.shape),), env.action_space.n)
    # Target Model (updated every 100 steps)
    target_model = agent((np.prod(env.S.shape),), env.action_space.n)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    steps_to_update_target_model = 0
    total_training_rewards = 0
    for episode in range(train_episodes):
        logger.debug(f"Starting episode {episode}")
        observation, _ = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1

            random_number = np.random.rand()
            encoded = observation
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = env.action_space.sample(env.valid_actions())
            else:
                predicted = model.predict(encoded, verbose=0)
                valid_actions = env.valid_actions()
                predicted = predicted * valid_actions #apply valid action mask
                action = _get_action_from_predicted_arr(predicted)
            new_observation, reward, done, info = env.step(action)
            replay_memory.append([observation, action, reward, new_observation, done])

            observation = new_observation
            total_training_rewards += reward
        logger.debug(f"Finished episode {episode}")
        # 3. Update the Main Network using the Bellman Equation
        if steps_to_update_target_model >= 50:
            logger.debug(f"Training model in episode {episode}")
            train(replay_memory, model, target_model)
            logger.debug(f"Finished training model in episode {episode}")
        # if episode % 5 == 0:
        #     train(replay_memory, model, target_model)

            logger.debug('Copying main network weights to the target network weights')
            target_model.set_weights(model.get_weights())
            steps_to_update_target_model = 0
            logger.debug(f"Finished copying main network weights to the target network weights in episode {episode}")
        if episode % 30 == 0:
            logger.info('Total training rewards: {} after {} episodes with latest reward = {}, epsilon = {}'.format(
                total_training_rewards, episode, reward, epsilon))
            total_training_rewards = 0

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
    env.close()


if __name__ == '__main__':
    main()
