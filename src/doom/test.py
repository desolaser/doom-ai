from vizdoom import *
import torch
import torch.nn as nn
import pickle
import itertools as it
from functools import partial
from doom.train_dqn import test as DQNTest
from doom.train_daqn import test as DAQNTest
from doom.train_dqnstack import test as DQNStackTest
from model.dqn import dqn
from model.dqnstack import dqnstack
from model.qnet import qnet
from model.autoencoder import autoencoder

def initialize_vizdoom(config_path):
	""" Initialize vizdoom game class an set configuration

	Keyword arguments: 
	config_path -- path of .cfg file used to get the maps and gameplay configuration
	"""

	print("Initializing doom...")
	game = DoomGame()
	game.load_config(config_path)
	game.set_window_visible(True)
	game.set_mode(Mode.ASYNC_PLAYER)
	game.set_screen_format(ScreenFormat.CRCGCB)
	game.set_screen_resolution(ScreenResolution.RES_400X225)
	game.init()
	print("Doom initialized.")
	return game
	
def test(config):

	config.game = initialize_vizdoom(config.config_path)
	n = config.game.get_available_buttons_size()	
	config.actions = [list(a) for a in it.product([0, 1], repeat=n)]
	
	if not config.load_model:		
		print('Model not defined')
	else:
		""" This code doesn't work in pytorch 1.0.0, delete it if you use that version. """

		pickle.load = partial(pickle.load, encoding="latin1")
		pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
		config.agent = torch.load(config.load_model, map_location=lambda storage, loc: storage, pickle_module=pickle)

		""" The code works in pytorch 1.0.0, if you use this version then uncomment the code below. """
		# config.model = torch.load(config.model_to_load)
		if config.model == 'dqn':
			config.agent.cuda()		
			DQNTest(config)
			return print('Testing finished.')

		elif config.model == 'dqnstack':
			config.resolution = (4, config.resolution[1], config.resolution[2])		
			config.agent.cuda()
			DQNStackTest(config)
			return print('Testing finished.')

		elif config.model == 'daqn':
			config.autoencoder = torch.load('../autoencoder.pth')
			config.agent.cuda()
			DAQNTest(config)
			return print('Testing finished.')

		else:
			print('Model '+config.model+' does not exists.')


		