import random
import numpy as np
import pandas as pd
from helper_functions import *
from analysis_tools import *
from ship_mdp import *
import time
import sys
import signal

def Qlearning(mdp, gamma, lambd, iterations, test_set, queue, t_queue):
	'''
	Explain function
	'''

	# Calculator for best policy changes
	policy_change_calc = 0

	# Read some parameters from INI file
	progress_interval, qmap_save_interval, policy_stat_save_interval = getQlearnOptions()
	
	progress = 0

	# The Q-values are a real-valued dictionary Q[s,a] where s is a state and a is an action.
	Q = dict()
	# Initialize Q values as zero
	for s in mdp.grid:
		for a in mdp.applicableActions(s):
			Q[s,a] = Action(-500000)

  # s is the current state we are looking at
  # (force s to start at timestep 0, with eng 1 and 3 on,
  # because starting at zero would make little sense with
  # power demand already there)
	s = dict2tuple(mdp.state)

	for i in range(iterations):

		# Get a value for epsilon, which will be used in case the e-greedy method is used
		epsilon = epsilonSelector(i, iterations)

		# save policy stats
		if (i % policy_stat_save_interval == 0) and (i != 0):
			print("saving stats")
			saveStats(Q, mdp, test_set, i, gamma, lambd, policy_change_calc)
			# reset policy change calculator
			policy_change_calc = 0

		# save q-mappings
		if (i % qmap_save_interval == 0) and (i != 0):
			print("saving q")
			saveQ(Q, mdp, i, gamma, lambd)

		# Choose an action to take at the curent state:
		a = mdp.actionSelector(Q, epsilon, mode="e-greedy")
		mdp.execute(a)
		Q[s,a].add_occurance()

		update = lambd * (mdp.stateValue(action=a) + \
		(gamma * mdp.valueOfBestAction(Q)))

		previous_best = mdp.bestAction(Q)

		# Calculate new Q
		Q[s,a].value = ((1 - lambd) * Q[s,a].value) + update

		if previous_best != mdp.bestAction(Q):
			policy_change_calc += 1

		# Update current state
		s = dict2tuple(mdp.state)

		# Reset MDP if current state is final
		if mdp.checkFinal() == True:
			# Save stats about current control
			mdp.reset()
			s = dict2tuple(mdp.state)
	# Save the Q values we ended up with
	saveQ(Q, mdp, i, gamma, lambd)
	saveStats(Q, mdp, test_set, i, gamma, lambd, policy_change_calc)
	