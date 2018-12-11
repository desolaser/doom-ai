import numpy as np
import skimage.color, skimage.transform
import numpy as np
from PIL import Image

def screenshot(state, path):
	""" Save an state as image

	Keyword arguments: 
	state -- state to save as image
	"""
	image_name = path + '/image_{}.jpg'.format(randint(0,100000000000000000)) 
	image_array = state[0, :, :, :]
	image_array = np.ascontiguousarray(image_array.transpose(1,2,0))
	img = Image.fromarray(image_array, 'RGB')
	img.save(image_name)    

def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	return x

def preprocess(resolution, img): 
	""" Resizes and change data type of the image."""

	img = skimage.transform.resize(img, resolution)
	img = img.astype(np.float32)	
	return img

def grey_preprocess(resolution, img): 
	""" Resizes and change data type of the image."""

	img = skimage.transform.resize(img, (1, resolution[1], resolution[2]))
	img = img.astype(np.float32)	
	return img


class Stack:
	def __init__(self, resolution):
		state_shape = (4, resolution[1], resolution[2])
		self.state = np.zeros(state_shape, dtype=np.float32)

	def add_state(self, state):
		for x in range(self.state.shape[0] - 1):
			aux = self.state[x, :, :]
			self.state[x + 1, :, :] = aux

		self.state[0, :, :] = state


