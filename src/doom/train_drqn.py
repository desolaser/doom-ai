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
from utils.image import preprocess
from tqdm import trange

def learn(criterion, agent, optimizer, hx, cx, s1, target_q):	
	""" Performs a learning step

	Keyword arguments:
	config -- imports global variables    
	s1 -- state received
	target_q -- target q values
	"""

	s1 = torch.from_numpy(s1)
	target_q = torch.from_numpy(target_q)

	if torch.cuda.is_available():
		hx, cx = hx.cuda(), cx.cuda()
		s1, target_q = s1.cuda(), target_q.cuda()

	hx, cx = Variable(hx), Variable(cx)
	s1, target_q = Variable(s1), Variable(target_q)
	output, hx, cx = agent(s1, hx, cx)

	loss = criterion(output, target_q)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

def get_q_values(agent, hx, cx, state):
	""" Returns q values of a given state

	Keyword arguments:
	config -- imports global variables    
	state -- given state
	"""

	state = torch.from_numpy(state)	

	if torch.cuda.is_available():
		hx, cx = hx.cuda(), cx.cuda()
		state = state.cuda()

	hx, cx = Variable(hx), Variable(cx)
	state = Variable(state)
	output, hx, cx = agent(state, hx, cx)
	return output

def get_best_action(agent, hx, cx, state):
	""" Get best action of a given state

	Keyword arguments: 
	state -- given state
	"""

	q = get_q_values(agent, hx, cx, state)
	m, index = torch.max(q, 1)
	action = index.cpu().data.numpy()[0]
	return action

def learn_from_memory(memory, batch_size, agent, discount_factor, 
					  criterion, optimizer, actual_epoch_loss_vector, hx, cx):
	""" Learns from a transition, that comes from the replay memory.
	s2 is ignored if isTerminal equals true """

	if memory.size > batch_size:
		s1, a, s2, isterminal, r = memory.get_sample(batch_size)   
		q = get_q_values(agent, hx, cx, s2).cpu().data.numpy() 
		q2 = np.max(q, axis=1)
		target_q = get_q_values(agent, hx, cx, s1).cpu().data.numpy()
		target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
		loss = learn(criterion, agent, optimizer, hx, cx, s1, target_q)
		actual_epoch_loss_vector.append(loss.cpu().detach().numpy())

def perform_learning_step(num_epochs, game, actions, resolution, frame_repeat, memory, batch_size, 
					  agent, discount_factor, criterion, optimizer, actual_epoch_loss_vector, hx, cx, epoch):
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
	
	s1 = preprocess(resolution, game.get_state().screen_buffer)
	eps = exploration_rate(num_epochs, epoch)

	if random() <= eps:
		a = randint(0, len(actions) - 1)
	else:		
		s1 = s1.reshape([1, resolution[0], resolution[1], resolution[2]])
		a = get_best_action(agent, hx, cx, s1)

	reward = game.make_action(actions[a], frame_repeat)
	isterminal = game.is_episode_finished()
	s2 = preprocess(resolution, game.get_state().screen_buffer) if not isterminal else None
	memory.add_transition(s1, a, s2, isterminal, reward)

	learn_from_memory(memory, batch_size, agent, discount_factor, 
		     criterion, optimizer, actual_epoch_loss_vector, hx, cx)

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
		config.actual_epoch_loss_vector = []  

		print("Training...")
		config.game.new_episode()

		hx = Variable(torch.zeros(config.batch_size, len(config.actions)).cuda())
		cx = Variable(torch.zeros(config.batch_size, len(config.actions)).cuda()) 

		for learning_step in trange(config.learning_steps_per_epoch, leave=False):

			perform_learning_step(config.epochs, config.game, config.actions, config.resolution, config.frame_repeat, 
								  config.memory, config.batch_size, config.agent, config.discount_factor, 
								  config.criterion, config.optimizer, config.actual_epoch_loss_vector, hx, cx, epoch)

			if config.game.is_episode_finished():
				score = config.game.get_total_reward()
				train_scores.append(score)
				config.game.new_episode()					
				train_episodes_finished += 1

		print("%d training episodes played." % train_episodes_finished)

		train_scores = np.array(train_scores)

		print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
			  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

		average_loss = sum(config.actual_epoch_loss_vector) / len(config.actual_epoch_loss_vector)
		config.loss_vector.append(average_loss)
		config.reward_vector.append(train_scores.mean())
		config.epoch_vector.append(epoch)

		trace = dict(x=config.epoch_vector, y=config.loss_vector, mode="markers+lines", 
				type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
		layout = dict(title="Loss function", xaxis={'title': 'epochs'}, yaxis={'title': 'loss'})
		config.vis._send({'data': [trace], 'layout': layout, 'win': 'losswin'})        

		print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))          


		trace = dict(x=config.epoch_vector, y=config.reward_vector, mode="markers+lines", 
				type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"})
		layout = dict(title="Testing reward graph", xaxis={'title': 'epochs'}, yaxis={'title': 'reward'})
		config.vis._send({'data': [trace], 'layout': layout, 'win': 'rewardwin'})

		save_file = config.model_path + "model_{}.pth".format(epoch)
		print("Saving the network weigths to:", save_file)
		torch.save(config.agent, save_file)         