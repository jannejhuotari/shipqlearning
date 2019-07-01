import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import csv
from ast import literal_eval as make_tuple
import configparser
from os import listdir
import seaborn as sns

# This class is used to store the values of state-action
# pairs and the amount of times the spesific action has
# been taken in that state.
class Action:
	'''
	Inputs to class Action:
		-value

	An action class is an object that is used to store state-action pairs
	for q-learning. The class contains variables "value" and "occurances",
	where value is of course the value of the state-action pair, and
	occurances stores how many times that action has been taken in
	previous iterations
	'''
	def __init__(self, value):
		self.value = value
		self.occurances = 0

	def add_occurance(self):
		self.occurances += 1

	def get_occurances(self):
		return self.occurances

def epsilonSelector(current_iteration, total_iterations):
	'''
	Inputs:
	current_iteration:		Current iteration in Q-Learn loop.
	total_iterations:		How many iterations there are in
							total in the Q-Learn loop.

	Outputs:
	 						Function outputs a value for epsilon,
							which gives the probability for selecting
							the least taken action in the e-greedy method.
							The probability is formulated so that it is high
							in the beginning of iterations, and it gets
							smaller as more of the state-space is explored.
	'''
	x = current_iteration / total_iterations

	return (np.e)**(-x)

def sfc_to_ffr(sfc, eng_power):
	'''
	Function converts given spesific fuel consumption points to
	fuel flow rate (from g/kwh in y axis and load % in x axis,
	to g/s in y axis and actual load in x axis). Current implementation
	only works for sfc points at 25%, 50%, 75%, 85% and 100%, and only
	on large marine diesel engines
	Input:
		-sfc: list of values for sfc, in the order
		  	  25%, 50%, 75%, 85% and 100%,
		-eng_power: max power of the engine in kW
	Returns:
		-a numpy poly1d object which can be called as output(power value)
	 	to get the fuel flow rate
	'''
	x_power = [eng_power*0.25, eng_power*0.5, eng_power*0.75, eng_power*0.85, eng_power]
	x_percentage = [25, 50, 75, 85, 100]

	# fuel flow rate from sfc points
	ffr = [(sfc[0]*x_power[0])/3600, (sfc[1]*x_power[1])/3600, (sfc[2]*x_power[2])/3600, (sfc[3]*x_power[3])/3600, (sfc[4]*x_power[4])/3600]

	# fit a polynomial to ffr:
	fit = np.polyfit(x_power, ffr, 2)

	return np.poly1d(fit)

def tuple2dict(state):
	dict_state = dict({
		'gen1': state[0], # 2400 kW gen
		'gen2': state[1], # 2400 kW gen
		'gen3': state[2], # 3200 kW gen
		'gen4': state[3], # 3200 kW gen
		'op_mode': state[4], # 1=maneuvring, 2=open sea, 3=archipelago, 4=port
		'distance_to_goal': state[5], # distance to goal
		'goal': state[6], # 0=port, 1=Mariehamn, 2=Stockholm, 3=Helsinki
		})
	return dict_state

def dict2tuple(state):
	return tuple([state['gen1'], state['gen2'], state['gen3'], state['gen4'],\
		state['op_mode'], state['distance_to_goal'], state['goal']])

def discretize(data, round_to):
	'''
	Function discretizes a given dataset by rounding
	each value with an accuracy of "round_to".
	Example:
	discretize([2, 6, 7, 9, 14], 5) would return:
	[0, 5, 5, 10, 15]

	Inputs:
	data:		data to discretize
	round_to:	discretization accuracy

	Output:
	dis_data:	discretized data
	'''
	dis_data = [0] * len(data)
	for i in range(len(data)):
		if data[i] % round_to >= round_to/2:
			dis_data[i] = data[i] + (round_to - (data[i] % round_to))
		else:
			dis_data[i] = data[i] - (data[i] % round_to)

	return dis_data

def createTrainingAndTestSets(spesific_set=1):
	'''
	Explain function
	'''

	# Trip number 5 has no trip to mariehamn on return so don't let
	# that get into the test sets.

	train_index = list(range(1, 10))
	test_index1 = random.choice(train_index)
	while test_index1 == 5:
		test_index1 = random.choice(train_index)

	train_index.remove(test_index1)
	test_index2 = random.choice(train_index)
	while test_index2 == 5:
		test_index2 = random.choice(train_index)
	train_index.remove(test_index2)

	test_index = [test_index1, test_index2]
	
	train_list = list()
	for i in train_index:
		train_list.append(pd.read_csv(\
			'datasets/trip_%d_full_dataset.csv' % i))

	train_set = pd.concat(train_list, sort=True)

	test_list = list()
	for i in test_index:
		test_list.append(pd.read_csv(\
			'datasets/trip_%d_full_dataset.csv' % i))

	test_set = pd.concat(test_list, sort=True)

	return train_set, test_set, pd.read_csv('datasets/trip_%d_full_dataset.csv' % spesific_set)

def createTroubleshootset(trip_number):
	'''
	Explain function
	'''

	train_set = pd.read_csv('datasets/trip_%d_full_dataset.csv'\
		% trip_number)

	test_set = pd.read_csv('datasets/trip_%d_full_dataset.csv'\
		% trip_number)

	return train_set, test_set

def readQ():
	
	gamma, lamda, weight, iteration, _, folder, _ = getAnalysisOptions()

	filename = "%s/qmap_gamma%.2f_lambda%.3f_weight%d_iteration%d.csv"\
	% (folder, gamma, lamda, weight, iteration)

	print("Found file: %s" % filename)

	Q = dict()

	with open(filename, 'r') as csvFile:
		reader = csv.reader(csvFile)
		for row in reader:
			Q[make_tuple(row[0]), row[1]] = Action(float(row[2]))

	csvFile.close()

	return Q

def getHyperparameters():
	config = configparser.ConfigParser()
	config.read('rlui.ini')

	gamma = float(config['hyperparameters']['gamma'])
	lamda = float(config['hyperparameters']['lambda'])
	weight = int(config['hyperparameters']['weight'])
	iterations = int(config['hyperparameters']['iterations'])
	return gamma, lamda, weight, iterations

def getQlearnOptions():
	config = configparser.ConfigParser()
	config.read('rlui.ini')

	progress_interval = float(config['qlearnOptions']['progress_interval'])
	qmap_save_interval = int(config['qlearnOptions']['qmap_save_interval'])
	policy_stat_save_interval = int(config['qlearnOptions']['policy_stat_save_interval'])
	return progress_interval, qmap_save_interval, policy_stat_save_interval

def getAnalysisOptions():
	config = configparser.ConfigParser()
	config.read('rlui.ini')

	gamma = float(config['analysisOptions']['analysis_gamma'])
	lamda = float(config['analysisOptions']['analysis_lambda'])
	weight = int(config['analysisOptions']['analysis_weight'])
	iterations = int(config['analysisOptions']['analysis_iterations'])
	trip_number = int(config['analysisOptions']['trip'])
	q_map_path = config['analysisOptions']['qmap_folder']
	policy_stats_path = config['analysisOptions']['qstats_folder']

	if iterations == 0:
		number = ''
		iterations_list = []
		files = listdir(q_map_path)
		for i in files:
			parts = i.split('_')
			parts[4] = parts[4].replace(".", "")
			for c in parts[4]:
				if c.isalpha() == False:
					number = number + c
			iterations_list.append(int(number))
			number = ''

		largest_iteration = 0
		for i in iterations_list:
			if i > largest_iteration:
				largest_iteration = i

		iterations = largest_iteration

	return gamma, lamda, weight, iterations, trip_number, q_map_path, policy_stats_path

def getParallelOptions():
	config = configparser.ConfigParser()
	config.read('rlui.ini')
	gammas = [0]*3
	lamdas = [0]*3
	weights = [0]*3

	gammas[0] = float(config['parallelOptions']['gamma_low'])
	gammas[1] = float(config['parallelOptions']['gamma_high'])
	gammas[2] = float(config['parallelOptions']['gamma_interval'])

	lamdas[0] = float(config['parallelOptions']['lambda_low'])
	lamdas[1] = float(config['parallelOptions']['lambda_high'])
	lamdas[2] = float(config['parallelOptions']['lambda_interval'])

	weights[0] = int(config['parallelOptions']['weight_low'])
	weights[1] = int(config['parallelOptions']['weight_high'])
	weights[2] = int(config['parallelOptions']['weight_interval'])

	iterations = int(config['parallelOptions']['iterations'])

	return gammas, lamdas, weights, iterations

def fuel_consumption_from_vectors(df, mdp):

	fc = 0
	for i in range(len(df)):

		# get current fuel consumption
		eng_powers = [df['g1'][i], df['g2'][i], df['g3'][i], df['g4'][i]]
		fc = fc + mdp.currentConsumption(eng_powers = eng_powers)

		if i > 0:

			# check gens started
			if df['g1'][i-1] == 0 and df['g1'][i] == 1:
				fc = fc + 50*180

			if df['g2'][i-1] == 0 and df['g2'][i] == 1:
				fc = fc + 50*180

			if df['g3'][i-1] == 0 and df['g3'][i] == 1:
				fc = fc + 50*180

			if df['g4'][i-1] == 0 and df['g4'][i] == 1:
				fc = fc + 50*180

			# check gens stopped
			if df['g1'][i-1] == 1 and df['g1'][i] == 0:
				fc = fc + 50*300

			if df['g2'][i-1] == 1 and df['g2'][i] == 0:
				fc = fc + 50*300

			if df['g3'][i-1] == 1 and df['g3'][i] == 0:
				fc = fc + 50*300

			if df['g4'][i-1] == 1 and df['g4'][i] == 0:
				fc = fc + 50*300

	# fc is in grams, return in tons:
	fc = fc / (1000*1000)
	return fc
