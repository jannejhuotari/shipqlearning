import itertools
import random
import numpy as np
import pandas as pd
from helper_functions import *

# ship_mdp contains two classes: shipMDP and Action.

class shipMDP:
	'''
 	Explain class
	'''
	def __init__(self, df, reserve_weight):

		# Zero means generator off, one means on
		self.g1_on = np.array([0, 1])
		self.g2_on = np.array([0, 1])
		self.g3_on = np.array([0, 1])
		self.g4_on = np.array([0, 1])
		self.op_mode = np.array([1, 2, 3, 4])
		self.distance_to_goal = np.arange(0, 400, 5)
		self.goal = np.array([0, 1, 2, 3])

		self.grid = list(itertools.product(self.g1_on,\
		self.g2_on, self.g3_on, self.g4_on, self.op_mode,\
		self.distance_to_goal, self.goal))

		# Initialize first state with a small and large gen set on
		self.state = dict({
			'gen1': 1, # 2400 kW gen
			'gen2': 0, # 2400 kW gen
			'gen3': 1, # 3200 kW gen
			'gen4': 0, # 3200 kW gen
			'op_mode': 4, # 1=maneuvring, 2=open sea, 3=archipelago, 4=port
			'distance_to_goal': 0, # distance to goal
			'goal': 0, # 0=port, 1=Mariehamn, 2=Stockholm, 3=Helsinki
			})

		self.timestep = 0

		self.power_demand = list(df['total_power'])
		self.power_reserve = list(df['reserve'])
		self.op_mode = list(df['operational_mode'])
		self.distance_to_goal = discretize(list(df['distance_to_goal']), 5)
		self.goal = list(df['goal'])

		self.last_step = len(self.power_demand)

		# Create fuel flow rate functions for engines:
		self.sfc = [210, 197, 190, 183, 186]

		# get fuel flow rate for state, multiply by 60 seconds
		# to get whole fuel constumption in state
		self.ffr12 = sfc_to_ffr(self.sfc, 2400) * 60
		self.ffr34 = sfc_to_ffr(self.sfc, 3200) * 60

		# State reward calculation weight
		self.reserve_weight = reserve_weight

	def applicableActions(self, state = None):
		'''
		applicableActions(self, state) takes a state (for ex. 
		(0, 0, 0, 0, 0, 2) as input) and return a list of 
		possible actions to take. Actions formatted as strings.
		'''

		actions = []
		if state == None:
			state = self.state
		else:
			# Convert given state to a dict representation
			state = tuple2dict(state)

		# Create a dummy list of engs on to check what happens if
		# we turn engines off
		engs_on = list(dict2tuple(state)[0:4])

		# Always possible to do nothing so append it
		actions.append("do_nothing")

		if state['gen1'] == 0:
			actions.append("1_on")
		else:
			engs_on[0] = 0
			if self.getLoadLevel(engs_on = engs_on) < 1 and \
			self.getLoadLevel(engs_on = engs_on) > 0:
				actions.append("1_off")

		engs_on = list(dict2tuple(state)[0:4])

		if state['gen2'] == 0:
			actions.append("2_on")
		else:
			engs_on[1] = 0
			if self.getLoadLevel(engs_on = engs_on) < 1 and \
			self.getLoadLevel(engs_on = engs_on) > 0:
				actions.append("2_off")

		engs_on = list(dict2tuple(state)[0:4])

		if state['gen3'] == 0:
			actions.append("3_on")
		else:
			engs_on[2] = 0
			if self.getLoadLevel(engs_on = engs_on) < 1 and \
			self.getLoadLevel(engs_on = engs_on) > 0:
				actions.append("3_off")

		engs_on = list(dict2tuple(state)[0:4])

		if state['gen4'] == 0:
			actions.append("4_on")
		else:
			engs_on[3] = 0
			if self.getLoadLevel(engs_on = engs_on) < 1 and \
			self.getLoadLevel(engs_on = engs_on) > 0:
				actions.append("4_off")


		return actions

	def leastSelectedAction(self, Q):
		'''
		Inputs:
		Q:		State-action pair values.
		Outputs:
		a:		Action taken the least amounts in current
				state.
		'''
		s = dict2tuple(self.state)
		actions = self.applicableActions()
		temp = Q[s, actions[0]].occurances
		a = actions[0]
		for k in actions:
			if Q[s,k].occurances < temp:
				temp = Q[s,k].occurances
				a = k
		if a == '':
			print("Empty action from leastSelectedAction\n")

		return a

	def bestAction(self, Q):
		'''
		Inputs:
		Q:			State-action pair values.
		Outputs:
		best_action:	Best action to take in the current state,
						according to Q.
		'''
		actions = self.applicableActions()
		a = ''
		value = -15000000
		for i in actions:
			if Q[dict2tuple(self.state), i].value > value:
				value = Q[dict2tuple(self.state), i].value
				a = i

		if a == '':
			print("Empty action from bestAction\n")

		return a

	def actionSelector(self, Q, epsilon, mode='exploration'):
		'''
		Inputs:
		Q:		State-action pair values.
		epsilon:	Probability of choosing the least selected action
				in state.
		mode: 	Either exploration or e-greedy. Choosing exploration
	 		 	will always return the least selected action taken
			 	in current state. e-greedy will select least selected
		 	 	action with probability epsilon, and the best action
			 	otherwise.
		Output:
		a:		Action
		'''
		a = ''
		# If selected mode is exploration, always choose the
		# action which has been taken the least amounts in
		# the current state
		if mode == "exploration":
			a = self.leastSelectedAction(Q)

		# If the selected mode is e-greedy, choose an action taken
		# the least in current state with the probability epsilon.
		# Otherwise, choose the best action according to Q.
		if mode == "e-greedy":
			if epsilon >= random.random():
				a = self.leastSelectedAction(Q)
			# else, choose the best action according to Q
			else:
				a = self.bestAction(Q)
		return a

	def execute(self, action):
		'''
		execute(self, action) performs action (given as a string). 
		In practice, it updates the classes state parameter according 
		to the action. The timestep variable of the state is always 
		incremented by 1
		'''

		self.timestep = self.timestep + 1

		if action == "1_on":
			self.state['gen1'] = 1

		elif action == "1_off":
			self.state['gen1'] = 0

		elif action == "2_on":
			self.state['gen2'] = 1

		elif action == "2_off":
			self.state['gen2'] = 0

		elif action == "3_on":
			self.state['gen3'] = 1

		elif action == "3_off":
			self.state['gen3'] = 0

		elif action == "4_on":
			self.state['gen4'] = 1

		elif action == "4_off":
			self.state['gen4'] = 0

		self.state['op_mode'] = self.op_mode[self.timestep]
		self.state['distance_to_goal'] = self.distance_to_goal[self.timestep]
		self.state['goal'] = self.goal[self.timestep]


	def getLoadLevel(self, engs_on = None):
		'''
		Function returns the total load level of all engines as a percentage
		in the range of 0 - 1.
		'''

		power_demand = self.power_demand[self.timestep]

		if engs_on == None:
			state = dict2tuple(self.state)
			# calculate amount of small engines online
			small_on = state[0:2].count(1)
			# same for large engines
			large_on = state[2:4].count(1)

		else:
			# calculate amount of small engines online
			small_on = engs_on[0:2].count(1)
			# same for large engines
			large_on = engs_on[2:4].count(1)

		# calculate total power available
		power_total = small_on * 2400 + large_on * 3200

		# calculate total load level
		if power_total != 0:
			load_level = (power_demand / power_total)
		else:
			load_level = 0

		return round(load_level, 3)

	def engPowers(self):
		'''
		engPowers(self)
		calculates the engine powers at current state. It is expected that 
		online engines operate at the same load percent, and they are 
		used just so that the power demand at that specific timestep is 
		fulfilled.
		'''
		# get current load level
		load_level = self.getLoadLevel()

		# calculate individual engine powers
		eng_powers = [0, 0, 0, 0]

		if self.state['gen1'] == 1:
			eng_powers[0] = 2400*load_level

		if self.state['gen2'] == 1:
			eng_powers[1] = 2400*load_level

		if self.state['gen3'] == 1:
			eng_powers[2] = 3200*load_level

		if self.state['gen4'] == 1:
			eng_powers[3] = 3200*load_level

		return eng_powers

	def checkBalance(self):
		'''
		Function checks if the current online gen-sets can fulfill
		the power demand.
		'''
		# get load level
		load_level = self.getLoadLevel()
		if (load_level > 1) or (load_level == 0):
			return False
		else:
			return True

	def checkReserve(self):
		'''
		Function checks if the current online gen-sets can fulfill
		the reserve power demand and power demand. If yes, returns -1,
		otherwise returns amount of reserve power missing.
		'''
		state = dict2tuple(self.state)
		# Calculate total power available
		# calculate amount of small engines online
		small_on = state[0:2].count(1)
		# same for large engines
		large_on = state[2:4].count(1)

		# calculate total power available
		power_total = small_on * 2400 + large_on * 3200

		if power_total - \
		self.power_demand[self.timestep] < self.power_reserve[self.timestep]:

			return self.power_reserve[self.timestep] - \
			(power_total - self.power_demand[self.timestep])
		else:
			return -1

	def currentConsumption(self, eng_powers=None):
		'''
		Returns current fuel consumption in mdp:s state
		'''

		# create option to use this as a generic consumption checker
		if eng_powers == None:
			eng_powers = self.engPowers()

		f_con = 0

		for i in eng_powers[0:2]:
			if i > 0:
				f_con = f_con + self.ffr12(i)

		for i in eng_powers[2:4]:
			if i > 0:
				f_con = f_con + self.ffr34(i)

		return f_con

	def maxConsumption(self):
		'''
		Returns fuel consumption in current state as if all engines
		we on.
		'''
		load = self.getLoadLevel()

		return self.ffr12(2400*load)*2 + self.ffr34(3200*load)*2


	def stateValue(self, action = 'do_nothing'):
		'''
		Explain function
		'''
		# Initialize state value:
		value = 0

		# Give reward depending on how much fuel we save compared
		# to the maximum fuel usage at state

		
		value = value + (self.maxConsumption() - self.currentConsumption())
		
		# Give a small penalty in fuel consumption for starting up
		# a generator. Penalty is FFR of engine multiplied by three
		# minutes in seconds, as this is the general ramp up time
		# for the engine
		
		
		if action == '1_on' or action == "2_on" or action == "3_on" or action == "4_on":
			value = value - (50*180)

		if action == '1_off' or action == "2_off" or action == "3_off" or action == "4_off":
			value = value - (50*300)
		

		# check if powerbalance is achieved and penalize if not:
		if self.checkBalance() == False:
			value = value - 90000000
		else:
			value = value

		# check if power reserve requirement is fulfilled and
		# penalize if not

		
		if self.checkReserve() != -1:
			value = value - (self.checkReserve() * self.reserve_weight)
		else:
			value = value
		

		return value

	def valueOfBestAction(self, Q):
		'''
		Function takes the current state-action value container (Q)
		as input and returns the value of the most valuable action
		to take in state self.state
		'''

		return Q[dict2tuple(self.state), self.bestAction(Q)].value

	def checkFinal(self):
		'''
		Helper function to check if we are at the last timestep
		in the analyzed timeseries.
		'''
		if self.timestep == self.last_step-1:
			return True

		else:
			return False

	# Helper function to reset the MDP in case we are at
	# the last timestep.
	def reset(self):
		self.timestep = 0
		self.state = dict({
			'gen1': 1, # 2400 kW gen
			'gen2': 0, # 2400 kW gen
			'gen3': 1, # 3200 kW gen
			'gen4': 0, # 3200 kW gen
			'op_mode': 4, # 1=maneuvring, 2=open sea, 3=archipelago, 4=port
			'distance_to_goal': 0, # distance to goal
			'goal': 0, # 0=port, 1=Mariehamn, 2=Stockholm, 3=Helsinki
			})
