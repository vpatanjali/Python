#!/usr/bin/python
# -*- coding: utf-8 -*-

import sets, sys, random

def solveIt(inputData):
	print "called solveit"
    # Modify this code to run your optimization algorithm

    # parse the input
	lines = inputData.split('\n')

	firstLine = lines[0].split()
	nodeCount = int(firstLine[0])
	edgeCount = int(firstLine[1])

	edges = []
	for i in range(1, edgeCount + 1):
		line = lines[i]
		parts = line.split()
		edges.append((int(parts[0]), int(parts[1])))

	# build a trivial solution
	# every node has its own color
	#solution = range(0, nodeCount)
	solution = algorithm(edges,nodeCount)

	# prepare the solution in the specified output format
	outputData = str(max(solution)+1) + ' ' + str(0) + '\n'
	outputData += ' '.join(map(str, solution))

	return outputData

def greedy(edges,nodeCount,num_trials = 1):

	best_coloring = [nodeCount-1]*nodeCount
	#edges = [(min(edge),max(edge)) for edge in edges]
	neighbours = [[] for i in xrange(nodeCount)]
	
	#for i in xrange(nodeCount):
	#	neighbours[i] = [edge[1] for edge in edges if edge[0] == i] + [edge[0] for edge in edges if edge[1] == i]
	
	for edge in edges:
		neighbours[edge[0]].append(edge[1])
		neighbours[edge[1]].append(edge[0])
	
	print "Finished building neighbour lists"
	
	degrees = [(len(neighbours[i]),i) for i in xrange(nodeCount)]
	degrees.sort(reverse=True)
	
	for trial in xrange(num_trials):
		coloring = [-1]*nodeCount
		#
		for node in degrees:
			neighbour_colors = sets.Set([coloring[neighbour] for neighbour in neighbours[node[1]]])
			for color in xrange(nodeCount):
				if color not in neighbour_colors:
					coloring[node[1]] = color
					break
		if(max(coloring)<max(best_coloring)):
			best_coloring = coloring
		random.shuffle(degrees)
		
	return best_coloring

def Dsatur(edges,nodeCount,num_trials = 1):

	coloring = [nodeCount-1]*nodeCount
	#edges = [(min(edge),max(edge)) for edge in edges]
	neighbours = [[] for i in xrange(nodeCount)]
	
	#for i in xrange(nodeCount):
	#	neighbours[i] = [edge[1] for edge in edges if edge[0] == i] + [edge[0] for edge in edges if edge[1] == i]
	
	for edge in edges:
		neighbours[edge[0]].append(edge[1])
		neighbours[edge[1]].append(edge[0])
	
	print "Finished building neighbour lists"
	
	degrees = [(len(neighbours[i]),i,sets.Set([])) for i in xrange(nodeCount)]
	degrees.sort(reverse=True)
	coloring[degrees[0][1]] = 0
	
	#Updating the neighbour color lists
	inv_sort_order = [(degrees[i][1],i) for i in xrange(len(degrees))]
	inv_sort_order.sort()
	sort_order = [x[1] for x in inv_sort_order] #Current indexes of original elements...
	
	for neighbour in neighbours[degrees[0][1]]:
		degrees[sort_order[neighbour]][2].add(0)
	
	#Replacing degree with color degree
	
	degrees = [(len(degrees[i][2]),degrees[i][1],degrees[i][2]) for i in xrange(nodeCount)]
	degrees[0] = (-1,degrees[0][1],degrees[0][2])
	
	for i in xrange(nodeCount-1):
		
		degrees.sort(reverse=True)
		inv_sort_order = [(degrees[i][1],i) for i in xrange(len(degrees))]
		inv_sort_order.sort()
		sort_order = [x[1] for x in inv_sort_order] #Current indexes of original elements...
		
		for color in xrange(nodeCount):
			if color not in degrees[0][2]:
				break
		
		# Updating own color and color degree
		
		coloring[degrees[0][1]] = color
		degrees[0] = (-1,degrees[0][1],degrees[0][2])
		
		# Updating neighbour color degrees
		for neighbour in neighbours[degrees[0][1]]:
			degrees[sort_order[neighbour]][2].add(color)
			if degrees[sort_order[neighbour]][0] > -1:
				degrees[sort_order[neighbour]] = (len(degrees[sort_order[neighbour]][2]),degrees[sort_order[neighbour]][1],degrees[sort_order[neighbour]][2])

	return coloring
	
	
algorithm = Dsatur

if __name__ == '__main__':
	if len(sys.argv) > 1:
		fileLocation = sys.argv[1].strip()
		inputDataFile = open(fileLocation, 'r')
		inputData = ''.join(inputDataFile.readlines())
		inputDataFile.close()
		print solveIt(inputData)
	else:
		print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)'

