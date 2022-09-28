from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def lenet(input_shape, numClasses, activation="relu", filters=[()], fc_size=256, weightsPath=None):
	"""
	reference: https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/
	"""

	model = Sequential()
	first_block = True

	# create blocks of conv->activation->pool
	for filter in filters:
		if first_block:
			model.add(Conv2D(filter[0], filter[1], padding="same", activation=activation, input_shape=input_shape))
		else:
			model.add(Conv2D(filter[0], filter[1], padding="same", activation=activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		first_block=False

	# fully connected layer
	model.add(Flatten())
	model.add(Dense(fc_size, activation=activation))

	# output layer
	model.add(Dense(numClasses, activation='softmax'))

	# if weightsPath is set, load weights 
	# from a pre-trained network
	if weightsPath is not None:
		model.load_weights(weightsPath)

	model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

	return model
