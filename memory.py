from collections import namedtuple
import random

Experience = namedtuple('Experience',
						('states','actions','next_states','rewards'))


class ReplayMemory:
	def __init__(self, capacity, seed):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		self.seed = random.seed(seed)

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Experience(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		# Randomly sample a batch of experiences from memory"
		return random.sample(self.memory, batch_size)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

