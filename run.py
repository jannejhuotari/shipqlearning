from ship_mdp import *
from ship_qlearning import *
import helper_functions
import pandas as pd
import matplotlib.pyplot as plt
import sys

args = sys.argv

def runQLearning(queue=None, t_queue=None):

	# read hyperparameters from INI file:
	gamma, lamda, weight, iterations = getHyperparameters()

	'''
	Create training and test data sets. In case you dont have data,
	you can just pass a dummy DataFrame to the MDP wich has the following
	data:
	power demand, named 'total_power' in the dataframe,
	reserve power, named 'reserve',
	operational mode, named 'operational_mode',
	distance to goal, named 'distance_to_goal' and
	goal, named 'goal'.
	'''
	train_set, test_set, _ = createTrainingAndTestSets()

	mdp = shipMDP(train_set, weight)

	QMap = Qlearning(mdp, gamma, lamda, iterations, test_set, queue, t_queue)

# Separate function for parallel runs
def parallelQLearning(gamma, lamda, weight, iterations):

	train_set, test_set, _ = createTrainingAndTestSets()

	# Can't use queues in parallel runs
	queue = None
	t_queue = None

	mdp = shipMDP(train_set, weight)

	QMap = Qlearning(mdp, gamma, lamda, iterations, test_set, queue, t_queue)