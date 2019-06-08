# TODO: your agent here!
from keras import layers, models, optimizers
from keras import backend as K
"""Replay Buffer
Most modern reinforcement learning algorithms benefit from using a replay memory or buffer to store and recall experience tuples.
Test for git
Here is a sample implementation of a replay buffer that you can use:"""
import random
from collections import namedtuple, deque
from noise import OUNoise
import numpy as np
import csv
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)       
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high,width,height):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        #self.BOARD_SHAPE = (width,height)
        self.BOARD_SHAPE = (width, height, 1)
        self.stateShape = self.BOARD_SHAPE
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        #states ----> actions
        #states = layers.Input(shape= self.BOARD_SHAPE, name='states')
       # states = layers.Reshape((10, 10), input_shape=self.BOARD_SHAPE);
        #statest = layers.Reshape((10, 10), input_shape=self.BOARD_SHAPE);
        states = layers.Input(shape=(self.state_size,) , name='states')
        #states = layers.Reshape((10, 10), input_shape=self.stateShape)
        #states = layers.Input(shape=layers.Reshape((10, 10), input_shape=self.stateShape), name='states')
        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=64, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='softmax',
            name='raw_actions')(net)
        actions= raw_actions
        print('raw_actions')
        print(raw_actions)
        # Scale [0, 1] output for each action dimension to proper range
        #actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
        #    name='actions')(raw_actions)
        print(actions)
        print('actions')
        #with open('agent', 'w') as csvfile:
            #writer = csv.writer(csvfile)
            #writer.writerow(raw_actions)  
            #writer.writerow(actions)  
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr= 0.1)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
class Critic:
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size,width,height):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.BOARD_SHAPE = (width,height,1)
        self.stateShape = self.BOARD_SHAPE
        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
       
        #states = layers.Input(shape=self.BOARD_SHAPE, name='states')
        #statest = layers.Reshape((10, 10), input_shape=self.BOARD_SHAPE);
        states = layers.Input(shape=(self.state_size,) , name='states')
        #states =ks.layers.Reshape((10, 10), input_shape=self.stateShape)
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, engine):
        self.task = engine
        self.width = engine.width
        self.height = engine.height
        self.state_size = engine.state_size
        self.action_size = engine.action_size
        self.action_low = engine.action_low
        self.action_high = engine.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,self.width,self.height)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,self.width,self.height)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size,self.width,self.height)
        self.critic_target = Critic(self.state_size, self.action_size,self.width,self.height)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 100
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.9  # discount factor
        self.tau = 0.05  # for soft update of target parameters
    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)        # Learn, if enough samples are available in memory
       # print(self.last_state)
        #print(action)
        #print(reward)
        #print(next_state)
        #print(done)
        #print('----')
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        #print(self.state_size)
        #state = np.reshape(state, [1, self.state_size])
        #print(state.shape)
        #print('act')
        action = self.actor_local.model.predict(state.reshape(1,self.state_size))[0]
        action1 = self.actor_local.model.predict(state.reshape(1,self.state_size))
        print('action1')
        print(action1)
        #action = action.squeeze(0).argmax()
        return list(action) 
        #return list(action + self.noise.sample())  # add some noise for exploration
    def act1(self, state):
        """Returns actions for given state(s) as per current policy."""
        #print(state)
        #print('act')
        state = np.reshape(state, [-1, self.state_size])
        #print(state)
        #print('act')
        action = self.actor_local.model.predict(state.reshape(1,self.state_size))[0]
        #my_state.reshape(1, OBSERVATION_SPACE)
        #print(action)
        action = np.argmax(action)
        #print(action)
        return action
        #return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None]).reshape(-1, self.state_size)
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        #actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
      
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).reshape(-1, self.state_size)

        _states = []
        for i in range(0,99): 
            _states.append(self.getState2(states[i]))
        
        _next_states = []
        for i in range(0,99): 
            _next_states.append(self.getState2(next_states[i]))
        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        #print(next_states)
        #print('next_states')
        _states = states
        _next_states = next_states
        actions_next = self.actor_target.model.predict_on_batch(_next_states)
        #value = round(np.array(actions_next).argmax())
        #for x in range(0,5):
            #actions_next[x] =value
        Q_targets_next = self.critic_target.model.predict_on_batch([_next_states, actions_next])
        #file_output = 'data3.txt' 
        #with open(file_output, 'w') as csvfile:
            ##writer = csv.writer(csvfile)
           # writer.writerow(np.array(Q_targets_next) ) 
       
        #Q_targets1 = rewards + self.gamma * Q_targets_next * (1 - dones)
        Q_targets = rewards + self.gamma * Q_targets_next 
        #file_output = 'data3.txt' 
        #with open(file_output, 'w') as csvfile:
            #writer = csv.writer(csvfile)
            #writer.writerow(np.array(Q_targets1) ) 
            #writer.writerow(np.array(Q_targets) ) 
        #print(Q_targets.shape)
        #print(actions.shape)
        #print(Q_targets.shape)
        self.critic_local.model.train_on_batch(x=[_states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([_states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([_states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   
       

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model arameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
    def reset_episode(self):
        self.noise.reset()
        #state = self.task.clear()
        self.task.clear()
        #self.last_state = state
        self.last_state =self.task.getState()
        return self.last_state

    def getState2(self,state):
        c = np.zeros(shape=(self.height, self.width), dtype=np.bool)
        for i in range(0,self.height):
            for j in range(0,self.width):
                c[i][j] = state[j + self.width* i]
        #print('99999999999999999999999999')
        #print(c)
        return c