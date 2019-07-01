import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import csv
from helper_functions import *
from ship_mdp import *
import os
import re

def saveStats(Q, training_mdp, test_set, iteration, gamma, lambd, policy_changes):
	'''
	Explain function
	'''

	# Create a shipMDP for test data and traverse it with the best actions:
	mdp = shipMDP(test_set, training_mdp.reserve_weight)

	_, _, stat_save_interval = getQlearnOptions()

	reserve_power_violation_counter = 0
	power_demand_violation_counter = 0
	fuel_consumption = 0
	total_reward = 0

	timesteps = mdp.last_step

	for i in range(timesteps-1):
		if mdp.checkReserve() > 500:
			reserve_power_violation_counter += 1

		if mdp.checkBalance() == False:
			power_demand_violation_counter += 1

		fuel_consumption = fuel_consumption + mdp.currentConsumption()
		if mdp.bestAction(Q) == '1_on' or mdp.bestAction(Q) == '2_on' \
		or mdp.bestAction(Q) == '3_on' or mdp.bestAction(Q) == '4_on':
			fuel_consumption = fuel_consumption + 50*180

		if mdp.bestAction(Q) == '1_off' or mdp.bestAction(Q) == '2_off' \
		or mdp.bestAction(Q) == '3_off' or mdp.bestAction(Q) == '4_off':
			fuel_consumption = fuel_consumption + 50*300

		mdp.execute(mdp.bestAction(Q))

		total_reward = total_reward + \
		mdp.stateValue(action=mdp.bestAction(Q))

	row = [str(iteration), str(reserve_power_violation_counter), \
	str(power_demand_violation_counter), str(fuel_consumption), \
	str(total_reward), str(policy_changes)]

	if iteration == stat_save_interval:
		with open("qstats/qstats_reserve%dgamma%.2flambda%.3f.csv" %\
		(mdp.reserve_weight, gamma, lambd), 'w') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(['iteration', 'reserve_violation',\
				'demand_violation', 'fuel_consumption', 'total_reward',\
				'policy_changes'])
		csvFile.close()

	with open("qstats/qstats_reserve%dgamma%.2flambda%.3f.csv" %\
		(mdp.reserve_weight, gamma, lambd), 'a') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(row)

	csvFile.close()

def printStats(Q, training_mdp, test_set, iteration, gamma, lambd):
	'''
	Explain function
	'''

	# Create a shipMDP for test data and traverse it with the best actions:
	mdp = shipMDP(test_set, training_mdp.reserve_weight)

	reserve_power_violation_counter = 0
	power_demand_violation_counter = 0
	fuel_consumption = 0
	total_reward = 0

	timesteps = mdp.last_step

	for i in range(timesteps-1):
		if mdp.checkReserve() > 500:
			reserve_power_violation_counter += 1

		if mdp.checkBalance() == False:
			power_demand_violation_counter += 1

		fuel_consumption = fuel_consumption + mdp.currentConsumption()
		if mdp.bestAction(Q) == '1_on' or mdp.bestAction(Q) == '2_on' \
		or mdp.bestAction(Q) == '3_on' or mdp.bestAction(Q) == '4_on':
			fuel_consumption = fuel_consumption + 50*180

		if mdp.bestAction(Q) == '1_off' or mdp.bestAction(Q) == '2_off' \
		or mdp.bestAction(Q) == '3_off' or mdp.bestAction(Q) == '4_off':
			fuel_consumption = fuel_consumption + 50*300

		mdp.execute(mdp.bestAction(Q))

		total_reward = total_reward + \
		mdp.stateValue(action=mdp.bestAction(Q))

	print('''
	QMap stats:
	Reserve power violations over 500: %d
	Power demand violations: %d
	Fuel consumption: %d
	Total reward: %d''' % (reserve_power_violation_counter,\
		power_demand_violation_counter, fuel_consumption, \
		total_reward))

def saveQ(Q, mdp, iteration, gamma, lambd):
	for s in mdp.grid:
		for a in mdp.applicableActions(s):
			with open("qmaps/qmap_gamma%.2f_lambda%.3f_weight%d_iteration%d.csv"\
				% (gamma, lambd, mdp.reserve_weight, iteration), 'a') as csvFile:
				writer = csv.writer(csvFile)
				writer.writerow([s, a, Q[s,a].value])
			csvFile.close()

def create_vectors():

	print("Initializing analysis..\n")

	gamma, lamda, weight, _, trip, _, folder = getAnalysisOptions()

	filename = "%s/qstats_reserve%dgamma%.2flambda%.3f.csv"\
	% (folder, weight, gamma, lamda)

	print("Found file: %s" % filename)

	print('''
		Initializing analysis with hyperparameters:
		Gamma: %s
		Lambda: %s
		Reserve violation weight: %s
		''' % (gamma, lamda, weight))

	iteration = []
	reserve_violation = []
	demand_violation = []
	fuel_consumption = []
	reward = []
	policy_changes = []

	with open(filename, 'r') as csvFile:
		reader = csv.reader(csvFile)
		next(reader)
		for row in reader:
			iteration.append(row[0])
			reserve_violation.append(float(row[1]))
			demand_violation.append(float(row[2]))
			fuel_consumption.append(float(row[3]) / (1000*1000))
			reward.append(float(row[4]))
			policy_changes.append(float(row[5]))

	csvFile.close()

	Q = readQ()

	_, _, dataset = createTrainingAndTestSets(spesific_set=trip)

	mdp = shipMDP(dataset, weight)

	timesteps = mdp.last_step
	power_demand = mdp.power_demand

	g1 = [0] * timesteps
	g2 = [0] * timesteps
	g3 = [0] * timesteps
	g4 = [0] * timesteps
	total_power = [0] * timesteps
	reserve_power = [0] * timesteps

	for i in range(timesteps-1):
		g1[i] = mdp.engPowers()[0]
		g2[i] = mdp.engPowers()[1]
		g3[i] = mdp.engPowers()[2]
		g4[i] = mdp.engPowers()[3]

		small_on = dict2tuple(mdp.state)[0:2].count(1)
		large_on = dict2tuple(mdp.state)[2:4].count(1)
		total_power[i] = small_on * 2400 + large_on * 3200
		reserve_power[i] = total_power[i] - power_demand[i]
		mdp.execute(mdp.bestAction(Q))

	stat_df = pd.DataFrame({
		'iteration': iteration,
		'reserve_violations': reserve_violation,
		'demand_violations': demand_violation,
		'fuel_consumption': fuel_consumption,
		'cumulative_reward': reward,
		'policy_changes': policy_changes
		})

	control_df = pd.DataFrame({
		'power_demand': power_demand,
		'g1': g1,
		'g2': g2,
		'g3': g3,
		'g4': g4,
		})

	real_df = pd.DataFrame({
		'power_demand': dataset['total_power'],
		'g1': dataset['dg1_power'].values,
		'g2': dataset['dg4_power'].values,
		'g3': dataset['dg2_power'].values,
		'g4': dataset['dg3_power'].values
		})

	real_df = real_df.reset_index()

	control_consumption = fuel_consumption_from_vectors(control_df, mdp)
	real_consumption = fuel_consumption_from_vectors(real_df, mdp)

	control_df['control_consumption'] = [control_consumption] * len(control_df)
	control_df['real_consumption'] = [real_consumption] * len(control_df)

	print("Consumption with Q-learning: %s tons" % control_consumption)
	print("Consumption with real control: %s tons" % real_consumption)
	print("Percentage saved: %f" % ((1 - control_consumption/real_consumption) * 100))

	stat_df.to_csv("analysis/stat_dataframe.csv")
	control_df.to_csv("analysis/control_dataframe.csv")
	real_df.to_csv("analysis/real_control_dataframe.csv")

	plt.figure()
	plt.plot(reserve_violation)
	plt.title("Reseve violation development")
	plt.savefig("analysis/reserve_violation.png")

	plt.figure()
	plt.plot(demand_violation)
	plt.title("Demand violation development")
	plt.ylim(0, 4)
	plt.savefig("analysis/demand_violation.png")

	plt.figure()
	plt.plot(fuel_consumption)
	plt.title("Fuel consumption development [Tons]")
	plt.savefig("analysis/fuel_consumption.png")

	plt.figure()
	plt.plot(reward)
	plt.title("Cumulative reward development")
	plt.savefig("analysis/cumulative_reward.png")

	plt.figure()
	plt.plot(policy_changes)
	plt.title("Policy changes")
	plt.savefig("analysis/policy_changes.png")

	plt.figure()
	plt.plot(power_demand, label='Power demand')
	plt.plot(g1, label = "Gen 1 power")
	plt.plot(g2, label = "Gen 2 power")
	plt.plot(g3, label = "Gen 3 power")
	plt.plot(g4, label = "Gen 4 power")
	plt.legend()
	plt.title('Ship auxiliary network operation')
	plt.savefig("analysis/network_operation.png")

	plt.figure()
	plt.plot(real_df['power_demand'], label='Power demand')
	plt.plot(real_df['g1'], label = "Gen 1 power")
	plt.plot(real_df['g2'], label = "Gen 2 power")
	plt.plot(real_df['g3'], label = "Gen 3 power")
	plt.plot(real_df['g4'], label = "Gen 4 power")
	plt.legend()
	plt.title('Real ship auxiliary network operation')
	plt.savefig("analysis/real_network_operation.png")

