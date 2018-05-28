###
### model.py
###
### Description:
### This file is responsible for creating the convolutional neural network
### model to be used to classify sign language digit images.
###

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Activation, Dropout
from keras.models import Sequential
import constants as C

def create_model():
	"""Creates a CNN model.

	Returns
	-------
	Sequential
		The model with all the layers added onto it.
	"""

	model = Sequential()
	model.add(Conv1D(32, 3, activation='relu', input_shape=C.INPUT_SHAPE))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(64, 3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(128, 3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Conv1D(256, 3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(C.OUTPUT_COUNT, activation='softmax'))

	return model

def compile_model(model):
	"""Compiles the model

	Parameters
	----------
	model
		A Sequential model to compile
	"""

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def fit_model(model, X_train, Y_train):
	"""Fits the model

	Parameters
	----------
	model
		A Sequential model to train and fit
	X_train
		A Numpy array containing the training set
	Y_train
		A Numpy array containing the labels for the training set

	Returns
	-------
	history
		A History of the training of the given model
	"""

	return model.fit(X_train, Y_train, batch_size=C.BATCH_SIZE, epochs=C.EPOCHS, validation_split=0.1)

def evaluate_model(model, X_test, Y_test):
	"""Evaluates the model using the test set

	Parameters
	----------
	model
		A Sequential model to test against the test set
	X_test
		A Numpy array containing the test set
	Y_test
		A Numpy array containing the labels for the test set

	Returns
	-------
	metrics
		A list of metrics based from the evaluation of the test set
	"""

	return model.evaluate(X_test, Y_test, verbose=1)

