import click 
import torch
import utils.ae_trainer as AETrainer
import utils.replay as Replay
from doom.train import train as Trainer
from doom.test import test as Tester
from utils.config import Config

pass_config = click.make_pass_decorator(Config, ensure=True)

@click.group()
def cli():
	pass

@cli.command()
@click.option('--model', type=str, default='dqn', help='Model to use (dqn, daqn, drqn)')
@click.option('--learning_rate', type=float, default='0.00025', help='Learning rate of the neuronal network')
@click.option('--discount_factor', type=float, default='0.99', help='Discount factor of the Q-learning algorithm')
@click.option('--config_path', default='basic', help='Config .cfg filename')
@click.option('--load_model', type=str, help='Network model to load')

@pass_config
def train(config, model, learning_rate, discount_factor, config_path, 
	load_model):
	""" Train the model"""

	config.model = model
	config.learning_rate = learning_rate
	config.discount_factor = discount_factor
	config.config_path = '../scenarios/'+config_path+'.cfg'
	config.load_model = load_model
	Trainer(config)

@cli.command()
@click.option('--model', type=str, default='dqn', help='Model to use (dqn, daqn, drqn, human)')
@click.option('--config_path', default='basic', help='Config .cfg filename')
@click.option('--load_model', type=str, help='Network model to load')

@pass_config
def test(config, model, config_path, load_model):
	""" Test the model"""

	config.model = model
	config.config_path = '../scenarios/'+config_path+'.cfg'
	config.load_model = load_model
	Tester(config)

@cli.command()
@click.option('--load_model', is_flag=True, help='Load previous model?')
@click.option('--learning_rate', type=float, default='1e-3', help='Autoencoder learning rate')
@click.option('--epochs', type=int, default=10, help='Number of epochs of training') #se recomienda entrenar por 600 épocas
def autoencoder(load_model, learning_rate, epochs): 
	""" Performs the training of the autoencoder."""

	AETrainer.train(load_model, learning_rate, epochs)

@cli.command()
@click.option('--config_file', default='replay.cfg', help='.cfg config file')
@click.option('--episodes', type=int, default=1, help='Episodes to replay')
def recording(config_file, episodes): 
	""" Records the player and save the gameplay as lmp file, and saves the frames in the dataset."""

	Replay.recording(config_file, episodes)

@cli.command()
@click.option('--config_file', default='replay.cfg', help='.cfg config file')
@click.option('--replay_file', default='replay.lmp', help='.lmp replay file to watch')
def watchreplay(config_file, replay_file): 
	""" Replays an lmp file and saves all the frames in the dataset."""

	Replay.replay(config_file, replay_file)

if __name__ == '__main__':
    cli()
