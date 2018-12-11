from __future__ import division
from __future__ import print_function
from vizdoom import *
from random import choice, randint, random
import itertools as it
from time import time, sleep
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from utils.image import grey_preprocess, Stack
from tqdm import trange

def learn(criterion, agent, optimizer, s1, target_q):	
	""" Performs a learning step

	Keyword arguments:
	config -- imports global variables    
	s1 -- state received
	target_q -- target q values
	"""

	s1 = torch.from_numpy(s1)
	target_q = torch.from_numpy(target_q)
	if torch.cuda.is_available():
		s1, target_q = s1.cuda(), target_q.cuda()

	s1, target_q = Variable(s1), Variable(target_q)
	output = agent(s1)

	loss = criterion(output, target_q)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

def get_q_values(agent, state):
	""" Returns q values of a given state

	Keyword arguments:
	config -- imports global variables    
	state -- given state
	"""

	state = torch.from_numpy(state)	

	if torch.cuda.is_available():
		state = state.cuda()

	state = Variable(state)
	output = agent(state)

	return output

def get_best_action(agent, state):
	""" Get best action of a given state

	Keyword arguments: 
	state -- given state
	"""

	q = get_q_values(agent, state)
	m, index = torch.max(q, 1)
	action = index.cpu().data.numpy()[0]
	return action

def learn_from_memory(memory, batch_size, agent, discount_factor, 
					  criterion, optimizer, actual_epoch_loss_vector):
	""" Learns from a transition, that comes from the replay memory.
	s2 is ignored if isTerminal equals true """

	if memory.size > batch_size:
		s1, a, s2, isterminal, r = memory.get_sample(batch_size)    
		q = get_q_values(agent, s2).cpu().data.numpy() 
		q2 = np.max(q, axis=1)
		target_q = get_q_values(agent, s1).cpu().data.numpy()
		target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2

		loss = learn(criterion, agent, optimizer, s1, target_q)
		actual_epoch_loss_vector.append(loss.cpu().detach().numpy())

def perform_learning_step(stack, num_epochs, game, actions, resolution, frame_repeat, memory, batch_size, 
					  agent, discount_factor, criterion, optimizer, actual_epoch_loss_vector, epoch):
	""" Makes an action according to eps-greedy policy, observes the result
	(next state, reward) and learns from the transition.

	Keyword arguments: 
	config -- imports global variables    
	epoch -- actual epoch of learning
	"""

	def exploration_rate(num_epochs, epoch):
		""" Define exploration rate change over time

		Keyword arguments: 
		config -- imports global variables    
		epoch -- actual epoch of learning
		"""

		start_eps = 1.0
		end_eps = 0.1
		const_eps_epochs = 0.1 * num_epochs
		eps_decay_epochs = 0.6 * num_epochs

		if epoch < const_eps_epochs:
			return start_eps
		elif epoch < eps_decay_epochs:
			return start_eps - (epoch - const_eps_epochs) / \
							   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
		else:
			return end_eps
	
	s1 = grey_preprocess(resolution, game.get_state().screen_buffer)
	stack.add_state(s1)
	stack1 = stack.state

	eps = exploration_rate(num_epochs, epoch)
	if random() <= eps:
		a = randint(0, len(actions) - 1)
	else:		
		stack.state = stack.state.reshape([1, resolution[0], resolution[1], resolution[2]])
		a = get_best_action(agent, stack.state)

	reward = game.make_action(actions[a], frame_repeat)
	isterminal = game.is_episode_finished()

	s2 = grey_preprocess(resolution, game.get_state().screen_buffer) if not isterminal else None	
	stack.add_state(s2)
	stack2 = stack.state

	memory.add_transition(stack1, a, stack2, isterminal, reward)

	learn_from_memory(memory, batch_size, agent, discount_factor, 
					  criterion, optimizer, actual_epoch_loss_vector)

def train(config):

	print("Starting the training!")
	print("Learning rate: ", config.learning_rate)
	print("Discount factor: ", config.discount_factor)
	print("Epochs: ", config.epochs)
	print("Learning steps per epoch: ", config.learning_steps_per_epoch)
	print("Batch size: ", config.batch_size)

	time_start = time()

	for epoch in range(config.epochs):

		print("\nEpoch %d\n-------" % (epoch + 1))
		train_episodes_finished = 0		
		train_scores = []     
		stack = Stack(config.resolution)

		print("Training...")
		config.game.new_episode()
		
		for learning_step in trange(config.learning_steps_per_epoch, leave=False):

			perform_learning_step(stack, config.epochs, config.game, config.actions, config.resolution, config.frame_repeat, 
								  config.memory, config.batch_size, config.agent, config.discount_factor, 
								  config.criterion, config.optimizer, config.actual_epoch_loss_vector, epoch)
			if config.game.is_episode_finished():
				score = config.game.get_total_reward()
				train_scores.append(score)
				config.game.new_episode()					
				train_episodes_finished += 1

		print("%d training episodes played." % train_episodes_finished)

		train_scores = np.array(train_scores)

		print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
			  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

		if epoch % 1 == 0:
			print(stack.state.shape)
			save_image(torch.from_numpy(stack.state[0]).data, '../image0_{}.png'.format(epoch))
			save_image(torch.from_numpy(stack.state[1]).data, '../image1_{}.png'.format(epoch))
			save_image(torch.from_numpy(stack.state[2]).data, '../image2_{}.png'.format(epoch))
			save_image(torch.from_numpy(stack.state[3]).data, '../image3_{}.png'.format(epoch))

		average_loss = sum(config.actual_epoch_loss_vector) / len(config.actual_epoch_loss_vector)
		config.loss_vector.append(average_loss)
		config.epoch_vector.append(epoch)

		trace = dict(x=config.epoch_vector, y=config.loss_vector, mode="markers+lines", 
				type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
		layout = dict(title="Loss function", xaxis={'title': 'epochs'}, yaxis={'title': 'loss'})
		config.vis._send({'data': [trace], 'layout': layout, 'win': 'losswin'})         
		config.actual_epoch_loss_vector = []

		print("\nTesting...")
		test_episode = []
		test_scores = []
		for test_episode in trange(config.test_episodes_per_epoch, leave=False):
			config.game.new_episode()				
				
			while not config.game.is_episode_finished():
				state = grey_preprocess(config.resolution, config.game.get_state().screen_buffer)
				stack.add_state(state)
				stack.state = stack.state.reshape([1, config.resolution[0], config.resolution[1], config.resolution[2]])
				best_action_index = get_best_action(config.agent, stack.state)

				config.game.make_action(config.actions[best_action_index], config.frame_repeat)
			r = config.game.get_total_reward()
			test_scores.append(r)

		test_scores = np.array(test_scores)
		print("Results: mean: %.1f +/- %.1f," % (
			test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
			  "max: %.1f" % test_scores.max())

		save_file = config.model_path + "model_{}.pth".format(epoch)
		print("Saving the network weigths to:", save_file)
		torch.save(config.agent, save_file)         

		print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))          

		config.reward_vector.append(test_scores.mean())

		trace = dict(x=config.epoch_vector, y=config.reward_vector, mode="markers+lines", 
				type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
		layout = dict(title="Testing reward graph", xaxis={'title': 'epochs'}, yaxis={'title': 'reward'})
		config.vis._send({'data': [trace], 'layout': layout, 'win': 'rewardwin'})

	config.game.close()

def test(config):
	
	print("It's time to watch!")

	score_array = []
	score = 0

	sc_counter = 0

	for _ in range(config.episodes_to_watch):
		config.game.new_episode()				
		stack = Stack(config.resolution)	

		while not config.game.is_episode_finished():

			state = grey_preprocess(config.resolution, config.game.get_state().screen_buffer)	
			stack.add_state(state)
			stack.state = stack.state.reshape([1, config.resolution[0], config.resolution[1], config.resolution[2]])				
						
			best_action_index = get_best_action(config.agent, stack.state)
			config.game.set_action(config.actions[best_action_index])
			for _ in range(config.frame_repeat):
				config.game.advance_action()

		sleep(1.0)
		score = config.game.get_total_reward()
		print("Total score: ", score)
		score_array.append(score)

	average_score = sum(score_array) / 10
	print("Average score", average_score)	
