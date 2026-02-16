import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


class QNetwork(nn.Module):
	"""Neural Network for Q-Learning using PyTorch"""
	def __init__(self, state_size, action_size):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(state_size, 64)
		self.fc2 = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 8)
		self.fc4 = nn.Linear(8, action_size)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.relu(self.fc3(x))
		x = self.fc4(x)
		return x

# Alias for backward compatibility with older models
SimpleModel = QNetwork


class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size  # normalized previous days
		self.action_size = 3  # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval
		
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		
		# Set device (CPU for Windows compatibility)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Initialize model
		if is_eval and model_name:
			self.model = self._load_model("models/" + model_name)
		else:
			self.model = self._create_model()
		
		self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.criterion = nn.MSELoss()
	
	def _create_model(self):
		"""Create a new Q-Network"""
		return QNetwork(self.state_size, self.action_size)
	
	def _load_model(self, model_path):
		"""Load a saved PyTorch model"""
		model = QNetwork(self.state_size, self.action_size)
		model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
		model.eval()
		return model
	
	def act(self, state):
		"""Choose action using epsilon-greedy policy"""
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		
		# Convert state to tensor
		state_tensor = torch.FloatTensor(state).to(self.device)
		
		with torch.no_grad():
			q_values = self.model(state_tensor)
		
		return torch.argmax(q_values).item()
	
	def expReplay(self, batch_size):
		"""Train the model using vectorized batch processing for 100x speedup"""
		if len(self.memory) < batch_size:
			return

		# Randomly sample a batch of experiences
		mini_batch = random.sample(self.memory, batch_size)
		
		# Efficiently unpack and convert to tensors
		states = torch.FloatTensor(np.array([t[0] for t in mini_batch])).to(self.device).squeeze(1)
		actions = torch.LongTensor([t[1] for t in mini_batch]).to(self.device).view(-1, 1)
		rewards = torch.FloatTensor([t[2] for t in mini_batch]).to(self.device).view(-1, 1)
		next_states = torch.FloatTensor(np.array([t[3] for t in mini_batch])).to(self.device).squeeze(1)
		dones = torch.FloatTensor([1 if t[4] else 0 for t in mini_batch]).to(self.device).view(-1, 1)

		# Get current Q values for the actions taken
		current_q_values = self.model(states).gather(1, actions)

		# Compute the target Q values using Bellman equation
		with torch.no_grad():
			next_q_values = self.model(next_states).max(1)[0].view(-1, 1)
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

		# Calculate loss and perform single optimization step
		loss = self.criterion(current_q_values, target_q_values)
		
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		# Decay epsilon
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
	
	def save_model(self, filepath):
		"""Save the PyTorch model"""
		torch.save(self.model.state_dict(), filepath)


def load_model(model_path):
	"""Load a PyTorch model and return the state dict"""
	return torch.load(model_path, map_location=torch.device("cpu"), weights_only=False) 
