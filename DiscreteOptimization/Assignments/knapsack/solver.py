#!/usr/bin/python
# -*- coding: utf-8 -*-

import fractions, time, sys
sys.setrecursionlimit(20000)

global table, weights, values, capacity, max_value, best_solution, unit_values

MEM_LIMIT = 1000*1000*100

def solveIt(inputData,type=0):
	# Modify this code to run your optimization algorithm

	global table, weights, values, capacity
	# parse the input
	lines = inputData.split('\n')

	firstLine = lines[0].split()
	items = int(firstLine[0])
	capacity = int(firstLine[1])

	values = []
	weights = []

	for i in range(1, items+1):
		line = lines[i]
		parts = line.split()
		values.append(int(parts[0]))
		weights.append(int(parts[1]))
		
	solution = preProcessing(type)
	#solution = greedy(values, weights, capacity)
    # prepare the solution in the specified output format
	outputData = "%(value)s %(optimality)s \n" %(solution)
	outputData += ' '.join(map(str, solution['taken']))
	return outputData

def preProcessing(type):
	global table, weights, values, capacity
	num_items = len(values)
	
	# Handling cases where a simple DP strategy storing all possibilities would take too much space.
	"""
	if(num_items*capacity>MEM_LIMIT):
		print num_items, capacity
		#Trying a gcd approach
		weights_sorted = sorted(weights,reverse=True)
		gcd = weights_sorted[0]
		i = 1
		while(gcd > 1 and i < num_items):
			#print i
			gcd = fractions.gcd(gcd, weights_sorted[i])
			i += 1
		capacity = capacity/gcd
		print capacity
		weights = [x/gcd for x in weights]
		if(num_items*capacity < MEM_LIMIT):
			weights = [x/gcd for x in weights]
			solution = dynamicPrograming(values, weights, capacity)
		else:
			solution = dynamicProgramming2()
	else:
		solution = dynamicProgramming2()
	"""
	print num_items,capacity,MEM_LIMIT
	if(num_items*capacity > MEM_LIMIT and num_items <=100):
		print "b&b"
		return branchAndBound()
	elif(num_items*capacity < MEM_LIMIT):
		print "DP2"
		return dynamicProgramming2()
	else:
		print "greedy"
		return greedy()
	
	"""
	if type == 2:
		print "DP2"
		return dynamicProgramming2()
	if type == 1:
		print "DP1"
		return dynamicProgramming()
	if type == 0:
		print "b&b"
		return branchAndBound()
	"""
	
def dynamicProgramming():
	""" Using a dynamic programing approach to the knapsack problem """
	#print "Using DP1"
	global table, weights, values, capacity
	num_items = len(values)
	
	table = [[0]*(capacity+1) for j in xrange(num_items+1)] # table of optimal solutions for all n, w combinations in table[n][w]
	
	for n in xrange(1, num_items+1):
		for w in xrange(1, capacity+1):
			if(weights[n-1] <= w):
				table[n][w] = max(table[n-1][w], table[n-1][w-weights[n-1]] + values[n-1])
			else:
				table[n][w] = table[n-1][w]
	solution = backTrackDP(table, weights)
	
	return {'value':table[num_items][capacity], 'taken':solution, 'optimality':1}

def backTrackDP(table, weights):
	""" Backtracks the DP table to get the optimal solution """
	
	solution = [0]*(len(table)-1)
	
	current_weight = len(table[0]) - 1
	
	for n in xrange(len(table)-1,0,-1):
		if table[n][current_weight] != table[n-1][current_weight]:
			solution[n-1] = 1
			current_weight -= weights[n-1]
	
	return solution
	
def dynamicProgramming2():
	""" Using a recursive method to populate a global table"""
	#print "Using DP2"
	global table, weights, values, capacity
	table = {}
	num_items = len(values)
	for w in xrange(0, weights[0]):
		table[(1,w)] = 0
		table[(0,w)] = 0
	for w in xrange(weights[0], capacity + 1):
		table[(1,w)] = weights[0]
		table[(0,w)] = 0
	#for n in xrange(2, num_items+1):
	#	print n, len(table)
	#	populate(n, capacity)
	populate(num_items, capacity)	
	solution = backTrackDP2()	
	return {'value':table[(num_items,capacity)], 'taken':solution, 'optimality':1}
	
def populate(n,w):
	global table, weights, values, capacity
	if (n,w) not in table:
		if(weights[n-1] <= w):
			table[(n,w)] = max(populate(n-1,w-weights[n-1]) + values[n-1], populate(n-1,w))
		else:
			table[(n,w)] = populate(n-1,w)

	return table[(n,w)]

def backTrackDP2():
	""" Backtracks the DP table to get the optimal solution """
	global table, weights, values, capacity
	solution = [0]*(len(weights))
	
	current_weight = capacity
	
	for n in xrange(len(weights),0,-1):
		if populate(n,current_weight) != populate(n-1,current_weight):
			solution[n-1] = 1
			current_weight -= weights[n-1]
	
	return solution
		
def greedy():
	""" A trivial greedy algorithm for filling the knapsack it takes items in-order until the knapsack is full """
	global table, weights, values, capacity, max_value, best_solution, unit_values
	unit_values = [((values[i]*1.0)/weights[i],i,weights[i],values[i]) for i in xrange(len(weights))]
	unit_values.sort()
	inv_sort_order = [(unit_values[i][1],i) for i in xrange(len(unit_values))]
	inv_sort_order.sort()
	sort_order = [x[1] for x in inv_sort_order]
	
	weight = 0
	max_value = 0
	best_solution = [0]*len(weights)
	i = len(weights) - 1
	while(weight < capacity and i >= 0):
		if weight + unit_values[i][2] <= capacity:
			best_solution[i] = 1
			max_value += unit_values[i][3]
			weight += unit_values[i][2]
		else:
			best_solution[i]=0
		i -= 1

	best_solution = [best_solution[i] for i in sort_order]
	
	return {'value':max_value, 'taken':best_solution, 'optimality':0}
			
def branchAndBound():
	global table, weights, values, capacity, max_value, best_solution, unit_values
	unit_values = [((values[i]*1.0)/weights[i],i,weights[i],values[i]) for i in xrange(len(weights))]
	unit_values.sort()
	
	def calcMax(_capacity, n):
		global unit_values
		curr_capacity = _capacity
		value = 0
		i = len(unit_values) - 1
		while(curr_capacity>0 and i >= n):
			if(unit_values[i][2]<=curr_capacity):
				value += unit_values[i][3]
				curr_capacity -= unit_values[i][2]
			else:
				value += curr_capacity*unit_values[i][0]
				curr_capacity = 0
			i -= 1
		
		return value
		
		
	def bNb(curr_capacity,curr_value,curr_path,n):
		"""
		@param: n - Number of items used up till now (tree depth-1)
		"""
		global max_value, best_solution, capacity, unit_values
		
		if n == len(weights):
			if curr_value > max_value:
				max_value = curr_value
				best_solution = curr_path
			return None
		
		current_bound = calcMax(curr_capacity,n)
		
		if(current_bound + curr_value < max_value):
			return None
		
		bNb(curr_capacity,curr_value,curr_path + [0],n+1)
		if(unit_values[n][2]<=curr_capacity):
			bNb(curr_capacity-unit_values[n][2],curr_value+unit_values[n][3],curr_path+[1],n+1)

		
	max_value = -1
	current_value = 0
	current_capacity = capacity
	best_solution = []
	
	bNb(current_capacity,current_value,[],0)
	inv_sort_order = [(unit_values[i][1],i) for i in xrange(len(unit_values))]
	inv_sort_order.sort()
	sort_order = [x[1] for x in inv_sort_order]
	
	best_solution = [best_solution[i] for i in sort_order]
	
	return {'value':max_value, 'taken':best_solution, 'optimality':1}
	
import sys

if __name__ == '__main__':
	if len(sys.argv) > 1:
		
		fileLocation = sys.argv[1].strip()
		inputDataFile = open(fileLocation, 'r')
		outputFile = open(fileLocation + ".out", 'w')
		inputData = ''.join(inputDataFile.readlines())
		inputDataFile.close()
		outputFile.write(solveIt(inputData))
		"""		
		fileLocations = sys.argv[1:len(sys.argv)]
		fileLocations = [x.strip() for x in fileLocations]
		print fileLocations
		
	
		for inputDataFile in fileLocations:
			start1 = time.time()
			inputDataFile = open(inputDataFile, 'r')
			inputData = ''.join(inputDataFile.readlines())
			inputDataFile.close()
			output0 = solveIt(inputData,0)
			end1 = time.time()
			output1 = solveIt(inputData,1)
			end2 = time.time()
			output2 = solveIt(inputData,2)
			end3 = time.time()
			print end1-start1, end2-end1, end3-end2
			print output0
			print output1
			print output2
		"""
	else:
		print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'


		