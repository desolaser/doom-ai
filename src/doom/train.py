from vizdoom import *
import itertools as it
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import visdom
from functools import partial
from doom.replay_memory import ReplayMemory
from doom.train_dqn import train as DQNTrainer
from doom.train_daqn import train as DAQNTrainer
from model.dqn import dqn
from model.dqnstack import dqnstack
from model.qnet import qnet
from model.autoencoder import autoencoder
from utils.calculator import layerCalculator

def initialize_vizdoom(config_path):
	""" Initialize vizdoom game class an set configuration

	Keyword arguments: 
	config_path -- path of .cfg file used to get the maps and gameplay configuration
	"""

	print("Initializing doom...")
	game = DoomGame()
	game.load_config(config_path)
	game.set_window_visible(False)
	game.set_mode(Mode.PLAYER)
	game.set_screen_format(ScreenFormat.CRCGCB)
	game.set_screen_resolution(ScreenResolution.RES_400X225)
	game.init()
	print("Doom initialized.")
	return game

def train(config):

	config.game = initialize_vizdoom(config.config_path)
	n = config.game.get_available_buttons_size()	
	config.memory = ReplayMemory(config.resolution, config.replay_memory_size)  
	config.actions = [list(a) for a in it.product([0, 1], repeat=n)]
	config.criterion = nn.MSELoss()
	config.vis = visdom.Visdom()
	config.loss_vector = []  
	config.reward_vector = []
	config.epoch_vector = []
	config.actual_epoch_loss_vector = []

	if config.load_model:
		""" This code doesn't work in pytorch 1.0.0, delete it if you use that version. """

		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		config.agent = torch.load(config.load_model, map_location=lambda storage, loc: storage, pickle_module=pickle)

		""" The code works in pytorch 1.0.0, if you use this version then uncomment the code below. """
		# config.model = torch.load(config.model_to_load)
	else:
		if config.model == 'dqn':
			linear_input = layerCalculator(config.resolution[1], config.resolution[2])
			config.agent = dqn(len(config.actions), linear_input)
			config.agent.cuda()
			config.optimizer = torch.optim.SGD(config.agent.parameters(), config.learning_rate) 			
			DQNTrainer(config)
			return print('Training finished.')

		elif config.model == 'daqn':
			
			config.autoencoder = torch.load('../autoencoder.pth')
			config.agent = qnet(len(config.actions), config.code_size)
			config.agent.cuda()
			config.optimizer = torch.optim.SGD(config.agent.parameters(), config.learning_rate) 	
			DAQNTrainer(config)	
			return print('Training finished.')

		else:
			print('Model '+config.model+' does not exists.')
	