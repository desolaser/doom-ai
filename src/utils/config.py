import torch

class Config(object):
	"""Defines the configuration and global variables"""

	def __init__(self):
		self.epochs = 20
		self.learning_steps_per_epoch = 2000
		self.replay_memory_size = 10000
		self.batch_size = 32
		self.test_episodes_per_epoch = 100        
		self.frame_repeat = 4
		self.resolution = (3, 60, 108)
		self.episodes_to_watch = 10
		self.code_size = 1024
		self.sw = False
		self.frame_counter = 0
		self.model_path = '../nets/'
