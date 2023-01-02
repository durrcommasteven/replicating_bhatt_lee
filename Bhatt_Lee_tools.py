"""
here we define tools to help us implement Bhatt and Lee
"""

import numpy as np
from numpy import linalg as LA
from scipy.optimize import minimize
import functools


print("importing tools for replicating Bhatt and Lee")



"""
Functions to make matrices
"""

def tensorproduct(*M, use_sparse = False):
		if not use_sparse:
				return functools.reduce(lambda x, y: np.kron(x, y), M)
		else:
				return functools.reduce(lambda x, y: sparse.kron(x, y), M)

def Sx():
	return np.array([[0, 1],[1, 0]])/2

def Sy():
	return np.array([[0, 1],[-1, 0]])/(2j)

def Sz():
	return np.array([[1, 0],[0, -1]])/2

def Splus():
	return np.array([[0, 1],[0, 0]])

def Sminus():
	return np.array([[0, 0],[1, 0]])

def ident():
	return np.array([[1, 0], [0, 1]])



def get_energies(j12, j13, j14, j23, j24, j34):
	"""
	Get lowest 4 energies of a 4 closter hamiltonian
	"""
	
	h12 = (j12/2)*(tensorproduct(Splus(), Sminus(), ident(), ident())
			   +tensorproduct(Sminus(), Splus(), ident(), ident())
			   +2*tensorproduct(Sz(), Sz(), ident(), ident()))
	
	h13 = (j13/2)*(tensorproduct(Splus(), ident(), Sminus(), ident())
			   +tensorproduct(Sminus(), ident(), Splus(), ident())
			   +2*tensorproduct(Sz(), ident(), Sz(), ident()))
	
	h14 = (j14/2)*(tensorproduct(Splus(), ident(), ident(), Sminus())
			   +tensorproduct(Sminus(), ident(), ident(), Splus())
			   +2*tensorproduct(Sz(), ident(), ident(), Sz()))
	
	h23 = (j23/2)*(tensorproduct(ident(), Splus(), Sminus(), ident())
			   +tensorproduct(ident(), Sminus(), Splus(), ident())
			   +2*tensorproduct(ident(), Sz(), Sz(), ident()))
	
	h24 = (j24/2)*(tensorproduct(ident(), Splus(), ident(), Sminus())
			   +tensorproduct(ident(), Sminus(), ident(), Splus())
			   +2*tensorproduct(ident(), Sz(), ident(), Sz()))
	
	h34 = (j34/2)*(tensorproduct(ident(), ident(), Splus(), Sminus())
			   +tensorproduct(ident(), ident(), Sminus(), Splus())
			   +2*tensorproduct(ident(), ident(), Sz(), Sz()))
	
	H = h12+h13+h14+h23+h24+h34
	
	w, _ = LA.eigh(H)
	
	return sorted(w)[:4]


"""
Now include a fifth point, p5, and find the effective new interactions
"""

def get_projected_energies(j12, j13, j14, j23, j24, j34, j51, j52, j53, j54):
	"""
	The goal of this function is to return the 8 lowest enegies of the 1,2,3,4,5 system
	
	This is done in 3 parts:
	built the initial hamiltonian (without 5) in order to retrieve the eigenvectors
	
	build the projection operator from these
	
	build the full hamiltonian (1,2,3,4, AND 5), and apply the projection operator 
	
	return the lowest 8 eigenvalues
	"""
	
	h12 = (j12/2)*(tensorproduct(Splus(), Sminus(), ident(), ident(), ident())
			   +tensorproduct(Sminus(), Splus(), ident(), ident(), ident())
			   +2*tensorproduct(Sz(), Sz(), ident(), ident(), ident()))
	
	h13 = (j13/2)*(tensorproduct(Splus(), ident(), Sminus(), ident(), ident())
			   +tensorproduct(Sminus(), ident(), Splus(), ident(), ident())
			   +2*tensorproduct(Sz(), ident(), Sz(), ident(), ident()))
	
	h14 = (j14/2)*(tensorproduct(Splus(), ident(), ident(), Sminus(), ident())
			   +tensorproduct(Sminus(), ident(), ident(), Splus(), ident())
			   +2*tensorproduct(Sz(), ident(), ident(), Sz(), ident()))
	
	h23 = (j23/2)*(tensorproduct(ident(), Splus(), Sminus(), ident(), ident())
			   +tensorproduct(ident(), Sminus(), Splus(), ident(), ident())
			   +2*tensorproduct(ident(), Sz(), Sz(), ident(), ident()))
	
	h24 = (j24/2)*(tensorproduct(ident(), Splus(), ident(), Sminus(), ident())
			   +tensorproduct(ident(), Sminus(), ident(), Splus(), ident())
			   +2*tensorproduct(ident(), Sz(), ident(), Sz(), ident()))
	
	h34 = (j34/2)*(tensorproduct(ident(), ident(), Splus(), Sminus(), ident())
			   +tensorproduct(ident(), ident(), Sminus(), Splus(), ident())
			   +2*tensorproduct(ident(), ident(), Sz(), Sz(), ident()))
	
	h51 = (j51/2)*(tensorproduct(Splus(), ident(), ident(), ident(), Sminus())
			   +tensorproduct(Sminus(), ident(), ident(), ident(), Splus())
			   +2*tensorproduct(Sz(), ident(), ident(), ident(), Sz()))
	
	h52 = (j52/2)*(tensorproduct(ident(), Splus(), ident(), ident(), Sminus())
			   +tensorproduct(ident(), Sminus(), ident(), ident(), Splus())
			   +2*tensorproduct(ident(), Sz(), ident(), ident(), Sz()))
	
	h53 = (j53/2)*(tensorproduct(ident(), ident(), Splus(), ident(), Sminus())
			   +tensorproduct(ident(), ident(), Sminus(), ident(), Splus())
			   +2*tensorproduct(ident(), ident(), Sz(), ident(), Sz()))
	
	h54 = (j54/2)*(tensorproduct(ident(), ident(), ident(), Splus(), Sminus())
			   +tensorproduct(ident(), ident(), ident(), Sminus(), Splus())
			   +2*tensorproduct(ident(), ident(), ident(), Sz(), Sz()))
	
	
	Htot = h12+h13+h14+h23+h24+h34+h51+h52+h53+h54
	
	ws, vs = LA.eigh(Htot)
	
	return sorted(ws)[:8]


def split_up_eigenvalues(eigenvalues):
	"""
	we'll now identify the three corresponding clusters, given 8 eigenvalues 

	the eigenvalues have either 2 or 4 nearly degenerate levels

	return 4 nearly degenerate level, [two other levels]
	"""



	"""
	The easiest way to do this is to sort, then go through possible 
	eigenvalues
	"""
	eigenvalues = sorted(eigenvalues)

	#Lets go through the possible levels

	#2, 2, 4
	levels1_1 = eigenvalues[:2]
	levels2_1 = eigenvalues[2:4]
	levels3_1 = eigenvalues[4:]

	#2, 4, 2
	levels1_2 = eigenvalues[:2]
	levels2_2 = eigenvalues[2:6]
	levels3_2 = eigenvalues[6:]

	#4, 2, 2
	levels1_3 = eigenvalues[:4]
	levels2_3 = eigenvalues[4:6]
	levels3_3 = eigenvalues[6:]

	"""
	Select the levels which are the 'tightest'
	"""

	std_1 = np.std(levels1_1)+np.std(levels2_1)+np.std(levels3_1)
	std_2 = np.std(levels1_2)+np.std(levels2_2)+np.std(levels3_2)
	std_3 = np.std(levels1_3)+np.std(levels2_3)+np.std(levels3_3)

	"""
	which is tightest
	"""
	tightest = min([(std_1, 1), (std_2, 2), (std_3, 3)])[-1]

	if tightest == 1:
		return [levels1_1, levels2_1, levels3_1]

	elif tightest == 2:
		return [levels1_2, levels2_2, levels3_2]

	else:
		return [levels1_3, levels2_3, levels3_3]


"""
Given 8 eigenvalues, we now expect to be able to cluster them as 

4, 2, 2

The following function does this
"""

def identify_levels(eigenvalues):
	"""
	we'll now identify the three corresponding clusters, given 8 eigenvalues 

	the eigenvalues have either 2 or 4 nearly degenerate levels

	return 4 nearly degenerate level, [two other levels]
	"""



	"""
	The easiest way to do this is to sort, then go through possible 
	eigenvalues
	"""
	eigenvalues = sorted(eigenvalues)

	#Lets go through the possible levels

	#2, 2, 4
	levels1_1 = eigenvalues[:2]
	levels2_1 = eigenvalues[2:4]
	levels3_1 = eigenvalues[4:]

	#2, 4, 2
	levels1_2 = eigenvalues[:2]
	levels2_2 = eigenvalues[2:6]
	levels3_2 = eigenvalues[6:]

	#4, 2, 2
	levels1_3 = eigenvalues[:4]
	levels2_3 = eigenvalues[4:6]
	levels3_3 = eigenvalues[6:]

	"""
	Select the levels which are the 'tightest'
	"""

	std_1 = np.std(levels1_1)+np.std(levels2_1)+np.std(levels3_1)
	std_2 = np.std(levels1_2)+np.std(levels2_2)+np.std(levels3_2)
	std_3 = np.std(levels1_3)+np.std(levels2_3)+np.std(levels3_3)

	"""
	which is tightest
	"""
	tightest = min([(std_1, 1), (std_2, 2), (std_3, 3)])[-1]

	if tightest == 1:
		return np.mean(levels3_1), sorted([np.mean(levels1_1), np.mean(levels2_1)])

	elif tightest == 2:
		return np.mean(levels2_2), sorted([np.mean(levels1_2), np.mean(levels3_2)])

	else:
		return np.mean(levels1_3), sorted([np.mean(levels2_3), np.mean(levels3_3)])


"""
Now we use these levels to find the effective couplings j35, j45
"""
def fit_to_values(eigenvalues, j34):
	"""
	given 8 eigenvalues, we identify the correspoinding clusters
	then we find the effective j35, j45, which give us those
	"""
	eigenvalues = np.sort(eigenvalues)
	
	levels_4, pair_of_levels = identify_levels(eigenvalues)
	levels_2_1, levels_2_2 = sorted(pair_of_levels)

	"""
	now use the analytic solution
	use notation from the notebook
	"""
	beta = levels_4 - levels_2_1
	delta = levels_4 - levels_2_2
	gamma = j34 

	term = 3*(3*beta**2 + 6*beta*gamma - 10*beta*delta - 9*gamma**2 + 6*gamma*delta + 3 * delta**2)
	term *= (term>0) #set to zero if negative
	term = np.sqrt(term)

	jnew1 = (beta-gamma+delta)/2 + term/6
	jnew2 = (beta-gamma+delta)/2 - term/6

	#if its negative, set to zero
	#jnew1 *= (jnew1>0)
	#jnew2 *= (jnew2>0)

	#now randomly assign one of these to be j35, one to be j45
	if np.random.rand()>0.5:
		j35, j45 = jnew1, jnew2
	else:
		j35, j45 = jnew2, jnew1
	
	#FOR TESTING
	
	val1 = (1/4)*(-2*np.sqrt(j34**2 - j34*j35 +j35**2 -j45*(j34+j35) +j45**2) -j34-j35-j45)
	val2 = (1/4)*(2*np.sqrt(j34**2 - j34*j35 +j35**2 -j45*(j34+j35) +j45**2) -j34-j35-j45)
	val3 = (j34+j35+j45)/4

	#print("TESTING")
	#print([0, delta, beta])
	#print([val1-val1, val2-val1, val3-val1])
	#print('check hereeee')
	
	#assert(j35>=0)
	#assert(j45>=0)

	return j35, j45 


def alternate_fit_to_values(eigenvalues, j34):
	"""
	AN ALTERNATIVE TO THE ABOVE FUNCTION
	this tries to minimize an error furnction to fit

	given 8 eigenvalues, we identify the correspoinding clusters
	then we find the effective j35, j45, which give us those
	"""
	eigenvalues = np.sort(eigenvalues)
	
	levels = np.array(identify_levels(eigenvalues))

	levels -= np.min(levels)
	
	def eigenvalue_function(js):
		"""
		There's for sure a better way to contrain these
		inputs to be positive
		"""
		j35, j45 = abs(js[0]), abs(js[1])
		
		"""
		these are in order from least to greatest
		"""

		val1 = (1/4)*(-2*np.sqrt(j34**2 - j34*j35 +j35**2 -j45*(j34+j35) +j45**2) -j34-j35-j45)
		val2 = (1/4)*(2*np.sqrt(j34**2 - j34*j35 +j35**2 -j45*(j34+j35) +j45**2) -j34-j35-j45)
		val3 = (j34+j35+j45)/4
		
		ws = [val1, val2, val3]
		scaled_w = np.array(ws) - min(ws)
		
		#we want to minimize the total distance to the levels we already found
		return scaled_w

	def eigenvalue_error(js):
		"""
		The error here
		"""
		scaled_w = eigenvalue_function(js)
		
		#I'll say its the sum of the differences divided by the norm of the target 
		return np.sum(np.abs(scaled_w-levels))/LA.norm(levels)
	
	"""
	It is important that the values we fit to have low error
	I'll therefore try nelder-mead first, and work through other methods 
	"""
	error_threshold = 0.01


	x0 = np.random.rand(2)
	res = minimize(eigenvalue_error, x0, method='nelder-mead')
	#a list of candidates: (couplings, accuracy)
	candidates = [(res.x, eigenvalue_error(res.x), 'nelder-mead')] 


	count = 0
	while candidates[-1][1]>0.025 and count < 9:
		print('')
		x0 = np.random.rand(2)
		res = minimize(eigenvalue_error, x0, method='nelder-mead')
		print('new error', eigenvalue_error(res.x))
		print('target', levels)
		print('curren', eigenvalue_function(res.x))
		count+=1
		print('count: ', count)
		candidates.append((res.x, eigenvalue_error(res.x)))

	#alternate methods
	methods = ['Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

	for method in methods:
		count = 0
		while count < 9:
			print('')
			x0 = np.random.rand(2)
			res = minimize(eigenvalue_error, x0, method=method)
			print('new error', eigenvalue_error(res.x))
			print('target', levels)
			print('curren', eigenvalue_function(res.x))
			count+=1
			print('count: ', count)
			candidates.append((res.x, eigenvalue_error(res.x), method))

	candidates.sort(key = lambda x: x[1])

	print(candidates)

	best_couplings = candidates[0][0]

	"""
	find better ways of getting a low error coupling


	
	
	"""

	print('target', levels)
	print('curren', eigenvalue_function(best_couplings))

	print(eigenvalue_error(best_couplings))
	assert(eigenvalue_error(best_couplings)<.025)

	print('new couplings: ',eigenvalue_function(best_couplings))
	print(eigenvalue_function(best_couplings))
	print(levels)
	
	return best_couplings


"""
Now we define functions to deal with clustering

"""


"""
Find the maximum coupling, and the points which have that coupling
"""
def max_coupling(points, point_dict):
	"""
	Now we find the 2 spins involving the strongest coupling. 
	"""
	#strength, then the two points with that connection
	max_coupling = [-10**8, None]
	
	for p in points:
		l = point_dict[(p[0], p[1])]
		"""
		maximum coupling here is l[0][-1]
		l[0] is the max coupling element
		l[0][-1] is the element gives the coupling
		"""
		if l:
			if max_coupling[0]<abs(l[0][-1]):
				max_coupling[0] = abs(l[0][-1])
				max_coupling[-1] = [(p[0], p[1]), l[0][0]]

	return max_coupling[0], max_coupling[-1]


"""
are the points in cluster actually a cluster
"""
def check_cluster(cluster, point_dict):
	"""
	given 4 points, this function checks whether its a good cluster
	i.e. each of the 4 points have connections to the other 3
	"""

	#First check if there are 4 unique points
	if not len(set(cluster)) == len(cluster):
		return False

	p0, p1, p2, p3 = cluster
	
	points0, _ = zip(*point_dict[p0])
	connection_01_bool = p1 in points0
	if not connection_01_bool:
		return False
	
	points1, _ = zip(*point_dict[p1])
	connection_12_bool = (p2 in points1) 
	connection_02_bool = (p2 in points0)
	if (not connection_12_bool) or (not connection_02_bool):
		return False
	
	points2, _ = zip(*point_dict[p2])
	connection_23_bool = (p3 in points2) 
	connection_13_bool = (p3 in points1)
	connection_03_bool = (p3 in points0)
	if (not connection_23_bool) or (not connection_13_bool) or (not connection_03_bool):
		return False
	
	return True


"""
given a cluster, what are the couplings
"""
def get_cluster_couplings(cluster, point_dict):
	"""
	return the couplings for 4 points in a cluster
	"""
	p0, p1, p2, p3 = cluster
	
	points0, J0s = zip(*point_dict[p0])
	J01 = J0s[points0.index(p1)] #if p1 in points0 else 0
	J02 = J0s[points0.index(p2)] #if p2 in points0 else 0
	J03 = J0s[points0.index(p3)] #if p3 in points0 else 0
	
	points1, J1s = zip(*point_dict[p1])
	J12 = J1s[points1.index(p2)] #if p2 in points1 else 0
	J13 = J1s[points1.index(p3)] #if p3 in points1 else 0
	
	points2, J2s = zip(*point_dict[p2])
	J23 = J2s[points2.index(p3)] #if p3 in points2 else 0

	#FOR DEBUGGING
	"""
	if 0 in [J01, J02, J03, J12, J13, J23]:
		# in case there are zeros here I want to see them
		print("J01, J02, J03, J12, J13, J23")
		print(J01, J02, J03, J12, J13, J23)
		print(' ')
	"""
	
	return J01, J02, J03, J12, J13, J23



"""
Now we define tools to enact the RG 
"""


def remove_point(points_to_remove, points, point_dict):
	"""
	remove p from points, point_dict
	
	return new version
	"""
	#remove from array
	new_points = np.array([point for point in points if (tuple(point) not in points_to_remove)])
	
	#remove from dictionary
	for p in points_to_remove:
		#check that the point is still in point_dict
		#otherwise we have a problem
		assert(p in point_dict)	

		connections, strengths = zip(*point_dict[p])

		"""
		first remove p as a neighbor from each of the points in connections
		"""
		for i, connection in enumerate(connections):
			if type(point_dict[connection])==tuple:
				#modify this so it is mutable
				point_dict[connection] = list(point_dict[connection])

			try:
				point_dict[connection].remove((p, strengths[i]))
			except:
				ps, ss = zip(*point_dict[connection])
				print('p in ps', p in ps)
				print('s in ss', strengths[i] in ss)
				print(ss)
				print(strengths[i])
				assert((p, strengths[i]) in point_dict[connection])


		#now delete p from the point dict
		del point_dict[p]
	
	return new_points, point_dict

def update_coupling(p1, p2, Jnew, point_dict):
	"""
	given points p1, p2, update their coupling to Jnew

	If p1, p2 arent connected, then add it in
	"""

	#assert(Jnew >= 0)
	assert(p1 != p2)

	points1, strengths1 = zip(*point_dict[p1])
	points2, strengths2 = zip(*point_dict[p2])

	strengths1 = list(strengths1)
	strengths2 = list(strengths2)

	if p2 in points1:
		#these points are already connected
		#check that this matches what we expect
		assert(p1 in points2)

		strengths1 = list(strengths1)
		strengths1[points1.index(p2)] = Jnew

		strengths2 = list(strengths2)
		strengths2[points2.index(p1)] = Jnew

	else: 
		#add this connection
		strengths1.append(Jnew)
		strengths2.append(Jnew)

		points1 = list(points1)
		points1.append(p2)

		points2 = list(points2)
		points2.append(p1)

	#now sort the strengths and recombine
	point_dict[p1] = sorted(zip(points1, strengths1), key = lambda x: -x[-1])
	point_dict[p2] = sorted(zip(points2, strengths2), key = lambda x: -x[-1])
	
	return point_dict

"""
Now run tests
"""
import TEST_Bhatt_Lee_tools