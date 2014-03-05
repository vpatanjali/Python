#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *

import sys
sys.path.append('/home/patanjali/lib/python2.7/site-packages/')

import math, random, threading, statusbars

MAX_REC_DEPTH = 2
LINEAR_LIMIT = 1000000000
CUBIC_LIMIT = round(pow(LINEAR_LIMIT,1.0/3))
QUAD_LIMIT = round(pow(LINEAR_LIMIT,0.5))

class workerThread (threading.Thread):
	"WIP"
	def __init__(self, name, points, recDepth, type):
		threading.Thread.__init__(self)
		self.points = points
		self.recDepth = recDepth
		self.type = type
		self.name = name
	def run(self):
		#print "Starting " + self.name
		return divideNconquer(self.points,self.recDepth,self.type)
		#print "Exiting " + self.name

def length(point1, point2):
	return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def objectiveFunction(points, solution=None):
	if(solution is None):
		solution = arange(len(points))
	obj = length(points[solution[-1]], points[solution[0]])
	for index in range(0, len(solution)-1):
		obj += length(points[solution[index]], points[solution[index+1]])
	return obj

def solveIt(inputData):
	# Modify this code to run your optimization algorithm

	# parse the input
	lines = inputData.split('\n')

	nodeCount = int(lines[0])

	points = []
	for i in range(1, nodeCount+1):
		line = lines[i]
		parts = line.split()
		points.append((float(parts[0]), float(parts[1])))

	solution = algorithm(points)
	# calculate the length of the tour
	obj = length(points[solution[-1]], points[solution[0]])
	
	for index in range(0, nodeCount-1):
		obj += length(points[solution[index]], points[solution[index+1]])

	# prepare the solution in the specified output format
	outputData = str(obj) + ' ' + str(0) + '\n'
	outputData += ' '.join(map(str, solution))
	##print len(points)
	return outputData
	
def greedy(points):
	nodeCount = len(points)
	if (nodeCount>QUAD_LIMIT):
		##print "Graph too big, returning default"
		return range(nodeCount)
	if (nodeCount < 2):
		return xrange(nodeCount)
	"Order n^2"
	solution = array([0]*len(points))
	travelled = array([False]*len(points))
	solution[0] = 0
	travelled[0] = True
	curr_point = 0
	sb = statusbars.StatusBar(len(points))
	for i in xrange(1,len(points)):
		next_point = 0
		best_dist_to_next_point = 'a'
		for j in xrange(1,len(points)):
			if(not travelled[j]):
				dist_to_next_point = length(points[j],points[curr_point])
				if(dist_to_next_point < best_dist_to_next_point):
					best_dist_to_next_point = dist_to_next_point
					next_point = j
		curr_point = next_point
		solution[i] = next_point
		travelled[next_point] = True
		sb.update(i)
	##print objectiveFunction(points,solution)
	return solution

def randomPick(points):
	solution = range(len(points))
	random.shuffle(solution)
	return solution

def divideNconquer(points, recDepth = 1, type = 1):
	nodeCount = len(points)
	if (nodeCount>QUAD_LIMIT/(MAX_REC_DEPTH*MAX_REC_DEPTH)):
		#print "Graph too big, returning default"
		return range(nodeCount)
	if (nodeCount < 3):
		return xrange(nodeCount)
	median_index = len(points)/2
	x_median = sorted([x[0] for x in points])[median_index]
	y_median = sorted([x[1] for x in points])[median_index]
	
	q = [0]*5
	q[1] = [points[i] for i in xrange(len(points)) if points[i][0] <= x_median and points[i][1] <= y_median]
	q[2] = [points[i] for i in xrange(len(points)) if points[i][0] > x_median and points[i][1] <= y_median]
	q[3] = [points[i] for i in xrange(len(points)) if points[i][0] > x_median and points[i][1] > y_median]
	q[4] = [points[i] for i in xrange(len(points)) if points[i][0] <= x_median and points[i][1] > y_median]
	
	q_ind = [0]*5
	q_ind[1] = [i for i in xrange(len(points)) if points[i][0] <= x_median and points[i][1] <= y_median]
	q_ind[2] = [i for i in xrange(len(points)) if points[i][0] > x_median and points[i][1] <= y_median]
	q_ind[3] = [i for i in xrange(len(points)) if points[i][0] > x_median and points[i][1] > y_median]
	q_ind[4] = [i for i in xrange(len(points)) if points[i][0] <= x_median and points[i][1] > y_median]
	
	q_sol = [0]*5
	
	if(recDepth == MAX_REC_DEPTH):
		for i in xrange(1,5):
			q_sol[i] = greedy(q[i])
	else:
		for i in xrange(1,5):
			q_sol[i] = divideNconquer(q[i],recDepth+1,getType(recDepth+1,i))
	
	if(type == 1):
		order = [1,2,3,4]
	if(type == 2):
		order = [1,4,3,2]
	if(type == 3):
		order = [3,2,1,4]
		
	solution = []
	for ind in order:
		solution += [q_ind[ind][x] for x in q_sol[ind]]
		
	return solution

def getType(recDepth,i):

	if recDepth == 2:
		if(i ==1):
			return 2
		elif(i == 2):
			return 1
		elif(i == 3):
			return 1
		elif(i == 4):
			return 3
	else:
		print 1/0
		
def twoOpt(points1,solution=None):
	# Local search using two opt
	nodeCount = len(points1)
	if (nodeCount>QUAD_LIMIT):
		#print "Graph too big, returning default"
		if solution is None:
			return range(nodeCount)
		else:
			return solution
	
	points = [x for x in points1]	
	if solution == None:
		solution = arange(nodeCount)
	else:
		points = [points1[i] for i in solution]
	sb = statusbars.StatusBar(nodeCount)
	for i in xrange(nodeCount-1):
		for j in xrange(i+2,nodeCount-1):
			k = i+1
			l = j+1
			if(length(points[i],points[j]) + length(points[k],points[l]) < length(points[i],points[k]) + length(points[j],points[l])):
				points[k:l] = points[j:i:-1]
				solution[k:l] = solution[j:i:-1]
		j = nodeCount-1
		k = i + 1
		l = 0
		if(length(points[i],points[j]) + length(points[k],points[l]) < length(points[i],points[k]) + length(points[j],points[l])):
			points[k:nodeCount] = points[j:i:-1]
			solution[k:nodeCount] = solution[j:i:-1]
		sb.update(i)
	return solution

def kOpt(points1, K = 2, solution=None):
	# Local search using k opt
	nodeCount = len(points1)
	if (nodeCount > round(pow(LINEAR_LIMIT,1.0/K))):
		#print "Graph too big, returning default"
		if solution is None:
			return range(nodeCount)
		else:
			return solution
	
	points = [x for x in points1]	
	if solution == None:
		solution = arange(nodeCount)
	else:
		points = [points1[i] for i in solution]
	sb = statusbars.StatusBar(nodeCount)
	for i in xrange(nodeCount-1):
		for j in xrange(i+2,nodeCount-1):
			k = i+1
			l = j+1
			if(length(points[i],points[j]) + length(points[k],points[l]) < length(points[i],points[k]) + length(points[j],points[l])):
				points[k:l] = points[j:i:-1]
				solution[k:l] = solution[j:i:-1]
		j = nodeCount-1
		k = i + 1
		l = 0
		if(length(points[i],points[j]) + length(points[k],points[l]) < length(points[i],points[k]) + length(points[j],points[l])):
			points[k:nodeCount] = points[j:i:-1]
			solution[k:nodeCount] = solution[j:i:-1]
		sb.update(i)
	return solution
	
def insertInBetween(points):
	nodeCount = len(points)
	if (nodeCount>QUAD_LIMIT):
		#print "Graph too big, returning default"
		return range(nodeCount)
	
	points_ind = [[points[i],i] for i in range(2,len(points))]
	solution = [0,1]
	ans = points[0:2]
	for i in xrange(len(points_ind)):
		insert_ind = 0
		insert_dist = 'a'
		for j in xrange(len(ans)):
			dist = length(ans[j],points_ind[0][0])+length(ans[i+1],points_ind[0][0]) - length(ans[j],ans[(j+1)%len(ans)])
			if(dist<insert_dist):
				insert_dist = dist
				insert_ind = j
		ans.insert(insert_ind+1,points_ind[0][0])
		solution.insert(insert_ind+1,points_ind[0][1])
		points_ind.pop(0)
	return solution

def insertNearestInBetween(points):
	"Pick the node nearest to the current cycle and insert at the best place"
	nodeCount = len(points)
	if (nodeCount>CUBIC_LIMIT):
		#print "Graph too big, returning default"
		return range(nodeCount)
	points_ind = [[points[i],i] for i in xrange(2,nodeCount)]
	solution = [0,1]
	ans = points[0:2]
	sb = statusbars.StatusBar(nodeCount)
	for i in xrange(nodeCount-2):
		# Finding the nearest node
		min_dist = 'a'
		nearest_new_node_ind = points_ind[0][1]
		for j in xrange(nodeCount-2-i):
			for k in xrange(i+2):
				if (length(points_ind[j][0],ans[k]) < min_dist):
					min_dist = length(points_ind[j][0],ans[k])
					nearest_new_node_ind = j
					
		insert_ind = 0
		insert_dist = 'a'
		for j in xrange(i+2):
			dist = length(ans[j],points_ind[nearest_new_node_ind][0])+length(ans[i+1],points_ind[nearest_new_node_ind][0]) - length(ans[j],ans[(j+1)%len(ans)])
			if(dist<insert_dist):
				insert_dist = dist
				insert_ind = j
		ans.insert(insert_ind+1,points_ind[nearest_new_node_ind][0])
		solution.insert(insert_ind+1,points_ind[nearest_new_node_ind][1])
		points_ind.pop(nearest_new_node_ind)
		sb.update(i)
	return solution	
	
def insertNearestInBetween_twoOpt(points):
	solution = insertNearestInBetween(points)
	solution = twoOpt(points,solution)
	return solution

def greedy_twoOpt(points):
	solution = greedy(points)
	solution = twoOpt(points,solution)
	return solution
	
def divideNconquer_twoOpt(points):
	solution = divideNconquer(points)
	solution = twoOpt(points,solution)
	return solution
	
def tryAllAlgos(points, verbose = True):
	if len(points) < 2:
		return range(len(points))
	best_solution = range(len(points))
	best_obj = objectiveFunction(points, best_solution)
	best_algo = 'linear'
	for algorithm in [greedy, insertInBetween, insertNearestInBetween]:
		solution_seed = algorithm(points)
		objective_seed = objectiveFunction(points, solution_seed)
		solution_twoOpt = None
		iter = 1
		while (sum(solution_twoOpt != solution_seed) > 0):
			if solution_twoOpt is not None:
				solution_seed = copy(solution_twoOpt)
			solution_twoOpt = twoOpt(points, solution_seed)
			objective_twoOpt = objectiveFunction(points, solution_twoOpt)
			print "\nIteration: ", iter, "Algorithm: ", algorithm.func_name, objective_seed, objective_twoOpt, "\n"
			iter += 1
		if (objective_twoOpt < best_obj):
			best_obj = objective_twoOpt
			best_solution = solution_twoOpt
			best_algo = algorithm.func_name
	if verbose:
		print best_algo, best_obj, "\n"
	return(best_solution)
	
def usingMST(points):
	eulerian = getEulerian(points)

def getEulerian(points):
	mst = MST(points)
	mst = mst + [(x[1],x[0]) for x in mst]
	for i in mst:
		pass
		
def MST(points):
	fullMat = [[length(points[i],points[j]) for j in xrange(len(points))] for i in xrange(len(points))]
	
def Delauny(points):
	pass
	
#algorithm = divideNconquer
#algorithm = twoOpt
#algorithm = greedy
#algorithm = greedy_twoOpt
#algorithm = divideNconquer_twoOpt
#algorithm = insertInBetween
#algorithm = insertNearestInBetween
#algorithm = insertNearestInBetween_twoOpt
algorithm = tryAllAlgos
	
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)'

