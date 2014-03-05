#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.extend(['/home/patanjali/lib/python2.7/site-packages/','/home/patanjali/SourceCodes/'])

import math, random, threading, statusbars

def solveIt(inputData):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = inputData.split('\n')

	parts = lines[0].split()
	warehouseCount = int(parts[0])
	customerCount = int(parts[1])

	warehouses = []
	for i in range(1, warehouseCount+1):
		line = lines[i]
		parts = line.split()
		warehouses.append((int(parts[0]), float(parts[1])))

	customerSizes = []
	customerCosts = []

	lineIndex = warehouseCount+1
	for i in range(0, customerCount):
		customerSize = int(lines[lineIndex+2*i])
		customerCost = map(float, lines[lineIndex+2*i+1].split())
		customerSizes.append(customerSize)
		customerCosts.append(customerCost)

	sol = algorithm(warehouses, customerSizes, customerCosts) 	
	used = sol[0]
	solution = sol[1]

	# calculate the cost of the solution
	obj = sum([warehouses[x][1]*used[x] for x in range(0,warehouseCount)])
	for c in range(0, customerCount):
		obj += customerCosts[c][solution[c]]

	# prepare the solution in the specified output format
	outputData = str(obj) + ' ' + str(0) + '\n'
	outputData += ' '.join(map(str, solution))

	return outputData
		
	# build a trivial solution

def greedy(warehouses, customerSizes, customerCosts):
	# pack the warehouses one by one until all the customers are served

	customerCount = len(customerSizes)
	warehouseCount = len(warehouses)

	solution = [-1] * customerCount
	capacityRemaining = [w[0] for w in warehouses]

	warehouseIndex = 0
	for c in range(0, customerCount):
		if capacityRemaining[warehouseIndex] >= customerSizes[c]:
			solution[c] = warehouseIndex
			capacityRemaining[warehouseIndex] -= customerSizes[c]
		else:
			warehouseIndex += 1
			assert capacityRemaining[warehouseIndex] >= customerSizes[c]
			solution[c] = warehouseIndex
			capacityRemaining[warehouseIndex] -= customerSizes[c]

	used = [0]*warehouseCount
	for wa in solution:
		used[wa] = 1
	return (used,solution)

def greedy2(warehouses, customerSizes, customerCosts):
	# assign customers to the warehouse with lowest cost within capacity limits

	customerCount = len(customerSizes)
	warehouseCount = len(warehouses)

	solution = [-1] * customerCount
	used = [0]*warehouseCount
	capacityRemaining = [w[0] for w in warehouses]
	
	sb = statusbars.StatusBar(customerCount)
	for c in xrange(0, customerCount):
		cost = [(customerCosts[c][i] + (1-used[i])*warehouses[i][1],i) for i in xrange(warehouseCount)]
		cost.sort()
		for w_ind in xrange(warehouseCount):
			if(capacityRemaining[cost[w_ind][1]]) >= customerSizes[c]:
				solution[c] = cost[w_ind][1]
				used[cost[w_ind][1]] = 1
				capacityRemaining[cost[w_ind][1]] -= customerSizes[c]
				sb.update(c)
				break
	print ""
	return (used,solution)

def greedy3(warehouses, customerSizes, customerCosts):
	# create warehouses with lowest cost including cost of servicing maximum customers 

	customerCount = len(customerSizes)
	warehouseCount = len(warehouses)

	# costs = [[]]*customerCount
	# for c in xrange(0, customerCount):
		# costs[c] = [(customerCosts[c][i] + (1-used[i])*warehouses[i][1],i) for i in xrange(warehouseCount)]
		# costs[c].sort()
		
	solution = [-1] * customerCount
	used = [0]*warehouseCount
	capacityRemaining = [w[0] for w in warehouses]
	
	sb = statusbars.StatusBar(customerCount)
	assignedCustomerCount = 0
	print(customerCount)
	while(assignedCustomerCount<customerCount):
		setup_costs = [warehouses[i][1]*(1-used[i]) for i in xrange(warehouseCount)] # Initializing additional setup cost
		customers_serviced = [0]*warehouseCount
		serviced_capacity = [0]*warehouseCount
		# Calculating the total cost of additional setup and assigning most valuable customers
		for w_ind in xrange(warehouseCount):
			capacity_remaining = capacityRemaining[w_ind]
			cust_costs = [(customerCosts[c][w_ind]/customerSizes[c],c) for c in xrange(customerCount) if solution[c] == -1]
			cust_costs.sort()
			servicing_cost = 0
			customers_serviced[w_ind] = []
			for cust in cust_costs:
				if(capacity_remaining==0):
					break
				if(customerSizes[cust[1]] <= capacity_remaining):
					customers_serviced[w_ind].append(cust[1])
					capacity_remaining -= customerSizes[cust[1]]
					servicing_cost += customerCosts[cust[1]][w_ind]
					serviced_capacity[w_ind] += customerSizes[cust[1]]
			setup_costs[w_ind] += servicing_cost
		print max(serviced_capacity)
		utilities = [(setup_costs[w_ind]/serviced_capacity[w_ind],w_ind) for w_ind in xrange(warehouseCount) if serviced_capacity[w_ind] > 0]
		utilities.sort()
		winner = utilities[0][1]
		for c in customers_serviced[winner]:
			solution[c] = winner
			capacityRemaining[winner] -= serviced_capacity[winner]
			used[winner] = 1
		assignedCustomerCount += len(customers_serviced[winner])
		print(assignedCustomerCount)
		sb.update(assignedCustomerCount)
	print ""
	return (used,solution)

def localSwapSearch(warehouses, customerSizes, customerCosts, used, solution):
	
	customerCount = len(customerSizes)
	warehouseCount = len(warehouses)

	capacityRemaining = [w[0] for w in warehouses]	
	used_count = [0]*warehouseCount
	for c in xrange(customerCount):
		capacityRemaining[solution[c]] -= customerSizes[c]
		used_count[solution[c]] += 1
	
	sb = statusbars.StatusBar(customerCount)
	for c in xrange(customerCount):
		for w_ind in xrange(warehouseCount):
			if(customerCosts[c][w_ind] + (1-used[w_ind])*warehouses[w_ind][1] < customerCosts[c][solution[c]] and customerSizes[c] <= capacityRemaining[w_ind]):
				print "Swapping %s with %s for customer %s" %(solution[c], w_ind, c)
				used_count[solution[c]] -= 1
				if(used_count[solution[c]] == 0):
					used[solution[c]] = 0
				solution[c] = w_ind
				capacityRemaining[w_ind] -= customerSizes[c]
				used_count[w_ind] += 1
		sb.update(c)
	return (used, solution)
				
def	greedySwapSearch(warehouses, customerSizes, customerCosts):
	ret = greedy2(warehouses, customerSizes, customerCosts)
	used = ret[0]
	solution = ret[1]
	return localSwapSearch(warehouses, customerSizes, customerCosts, used, solution)
	
import sys

algorithm = greedySwapSearch

if __name__ == '__main__':
	if len(sys.argv) > 1:
		fileLocation = sys.argv[1].strip()
		inputDataFile = open(fileLocation, 'r')
		inputData = ''.join(inputDataFile.readlines())
		inputDataFile.close()
		print 'Solving:', fileLocation
		print solveIt(inputData)
	else:
		print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/wl_16_1)'

