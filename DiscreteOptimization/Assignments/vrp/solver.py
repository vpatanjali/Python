#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, sys

sys.path.append('../tsp/')

import tsp

def length(customer1, customer2):
	return math.sqrt((customer1[1] - customer2[1])**2 + (customer1[2] - customer2[2])**2)

def solveIt(inputData):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = inputData.split('\n')

	parts = lines[0].split()
	customerCount = int(parts[0])
	vehicleCount = int(parts[1])
	vehicleCapacity = int(parts[2])
	depotIndex = 0

	customers = []
	for i in range(1, customerCount+1):
		line = lines[i]
		parts = line.split()
		customers.append((int(parts[0]), float(parts[1]),float(parts[2])))

	vehicleTours = tryAllAlgos(vehicleCount, vehicleCapacity, customers)
	obj = objectiveFunction(vehicleTours, customers)
	# prepare the solution in the specified output format
	outputData = str(obj) + ' ' + str(0) + '\n'
	for v in range(0, vehicleCount):
		outputData += str(depotIndex) + ' ' + ' '.join(map(str,vehicleTours[v])) + ' ' + str(depotIndex) + '\n'

	return outputData

def greedy1(vehicleCount, vehicleCapacity, customers):
	# build a trivial solution
	# assign customers to vehicles starting by the largest customer demands

	vehicleTours = []
	
	customerCount = len(customers)
	
	customerIndexs = set(range(1, customerCount))  # start at 1 to remove depot index
	
	for v in range(0, vehicleCount):
		# print "Start Vehicle: ",v
		vehicleTours.append([])
		capacityRemaining = vehicleCapacity
		while sum([capacityRemaining >= customers[ci][0] for ci in customerIndexs]) > 0:
			used = set()
			order = sorted(customerIndexs, key=lambda ci: -customers[ci][0])
			for ci in order:
				if capacityRemaining >= customers[ci][0]:
					capacityRemaining -= customers[ci][0]
					vehicleTours[v].append(ci)
					# print '   add', ci, capacityRemaining
					used.add(ci)
			customerIndexs -= used

	# checks that the number of customers served is correct
	assert sum([len(v) for v in vehicleTours]) == customerCount - 1
	return vehicleTours

def angle(customer, origin = (0,0)):
	radius2 = customer[1]**2+customer[2]**2
	if radius2 == 0:
		return 0
	if customer[1] >= 0:
		if customer[2] >= 0:
			return customer[2]**2/radius2
		else:
			return 3 + customer[1]**2/radius2
	else:
		if customer[2] >= 0:
			return 1 + customer[1]**2/radius2
		else:
			return 2 + customer[2]**2/radius2
	
def radial(vehicleCount, vehicleCapacity, customers):
	# assign customers to vehicles starting by the largest customer demands

	vehicleTours = []
	
	customerCount = len(customers)
	
	customerIndexs = set(range(1, customerCount))  # start at 1 to remove depot index
	
	# Translating origin and computing angles
	origin = (customers[0][1],customers[0][2])
	for i in xrange(customerCount):
		customers[i] = (customers[i][0], customers[i][1] - origin[0], customers[i][2] - origin[1])
	customerAngles = zip(range(customerCount), map(angle, customers))
	
	for v in range(0, vehicleCount):
		# print "Start Vehicle: ",v
		vehicleTours.append([])
		capacityRemaining = vehicleCapacity
		while sum([capacityRemaining >= customers[ci][0] for ci in customerIndexs]) > 0:
			used = set()
			order = sorted(customerIndexs, key=lambda ci: customerAngles[ci][0])
			for ci in order:
				if capacityRemaining >= customers[ci][0]:
					capacityRemaining -= customers[ci][0]
					vehicleTours[v].append(ci)
					# print '   add', ci, capacityRemaining
					used.add(ci)
			customerIndexs -= used
	
	#Backtracking using a greedy step if we cant fit all customers
	
	K = 1
	while (len(customerIndexs)>0 and K < vehicleCount):
		print "Backtracking", K, "/", vehicleCount, "th time"
		for v in range(vehicleCount-K, vehicleCount):
			customerIndexs.update(vehicleTours[v])
			vehicleTours[v] = []
			
		for v in range(vehicleCount-K, vehicleCount):
			capacityRemaining = vehicleCapacity
			while sum([capacityRemaining >= customers[ci][0] for ci in customerIndexs]) > 0:
				used = set()
				order = sorted(customerIndexs, key=lambda ci: -customers[ci][0])
				for ci in order:
					if capacityRemaining >= customers[ci][0]:
						capacityRemaining -= customers[ci][0]
						vehicleTours[v].append(ci)
						# print '   add', ci, capacityRemaining
						used.add(ci)
				customerIndexs -= used
		K += 1
		
	# checks that the number of customers served is correct
	all_used = set()
	for v in vehicleTours:
		all_used.update(v)
	if(len(all_used) != customerCount - 1):
		print len(customerIndexs), "Still throws errors"
		raise 
	return vehicleTours
	
def localTSP(vehicleTours, customers):
	for vt_ind in xrange(len(vehicleTours)):
		new_order = tsp.tryAllAlgos([(customers[i][1], customers[i][2]) for i in ([0] + vehicleTours[vt_ind])])
		vehicleTours[vt_ind] = [vehicleTours[vt_ind][i-1] for i in new_order[1:]]
	return vehicleTours

def tryAllAlgos(vehicleCount, vehicleCapacity, customers):
	
	best_solution = None
	best_obj = 'a'
	best_algo = None
	for algorithm in [greedy1, radial]:
		solution_seed = algorithm(vehicleCount, vehicleCapacity, customers)
		print solution_seed
		objective_seed = objectiveFunction(solution_seed, customers)
		solution_TSP = localTSP(solution_seed, customers)
		print solution_TSP
		objective_TSP = objectiveFunction(solution_TSP, customers)
		print "\n", algorithm.func_name, objective_seed, objective_TSP, "\n"
		if (objective_TSP < best_obj):
			best_obj = objective_TSP
			best_solution = solution_TSP
			best_algo = algorithm.func_name
	print best_algo, best_obj, "\n"
	return(best_solution)
	
def objectiveFunction(vehicleTours, customers, depotIndex = 0):	
	# calculate the cost of the solution; for each vehicle the length of the route
	obj = 0
	vehicleCount = len(vehicleTours)
	for v in range(0, vehicleCount):
		vehicleTour = vehicleTours[v]
		if len(vehicleTour) > 0:
			obj += length(customers[depotIndex],customers[vehicleTour[0]])
			for i in range(0, len(vehicleTour) - 1):
				obj += length(customers[vehicleTour[i]],customers[vehicleTour[i + 1]])
			obj += length(customers[vehicleTour[-1]],customers[depotIndex])
	return obj

import sys

if __name__ == '__main__':
	if len(sys.argv) > 1:
		fileLocation = sys.argv[1].strip()
		inputDataFile = open(fileLocation, 'r')
		inputData = ''.join(inputDataFile.readlines())
		inputDataFile.close()
		print 'Solving:', fileLocation
		print solveIt(inputData)
	else:

		print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)'

