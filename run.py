### run.py
###
### Description:
### This file is main file that creates and evaluates the model using the dataset.
###

from get_dataset import get_dataset
from model import create_model, compile_model, fit_model, evaluate_model
import constants as C
import matplotlib.pyplot as plt

def plot_training_history(history):
	"""Plots the training history of a model

	Parameters
	----------
	history
		A History object of a given model
	"""

	training_loss = history.history['loss']
	validation_loss = history.history['val_loss']
	epochs = range(1, C.EPOCHS + 1)

	plt.plot(epochs, training_loss, label='Training loss')
	plt.plot(validation_loss, label='Validation loss')
	plt.title('Training and validation loss for ' + str(C.EPOCHS) + ' epochs')
	plt.gca().set_xlabel('Epochs')
	plt.gca().set_ylabel('Loss')
	plt.legend()
	plt.tight_layout()
	
	plt.gcf().savefig('loss.png')

if __name__ == '__main__':
	# Load the dataset and split them into training and test sets
	X_train, X_test, Y_train, Y_test = get_dataset()

	# Create the model and compile it
	model = create_model()
	compile_model(model)

	print(model.summary())
	print()

	print('Training model...')
	training_history = fit_model(model, X_train, Y_train)
	print()

	print('Evaluating model...')
	metrics = evaluate_model(model, X_test, Y_test)
	print()

	print('Loss on test set is:', metrics[0])
	print('Accuracy on test set is:', metrics[-1])
	print()

	# Uncomment to see the plot of the training and validation losses (loss.png)
	# print('Plotting training history...')
	# plot_training_history(training_history)
	# print('Done')
