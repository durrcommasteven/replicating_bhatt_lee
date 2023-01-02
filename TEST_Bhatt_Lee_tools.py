from Bhatt_Lee_tools import *

print('running checks on Bhatt_Lee_tools')

"""

Here we run some basic tests on Bhatt_Lee_tools to 
confirm that nothing is broken

"""
def TEST_get_projected_energies():
	couplings = np.random.rand(10)
	couplings[0]=100

	j12, j13, j14, j23, j24, j34, j51, j52, j53, j54 = couplings
	eigenvalues = get_projected_energies(j12, j13, j14, j23, j24, j34, j51, j52, j53, j54)

	"""
	lets see if the grouping we think is best, actually is best
	do this by looking at the sum of the stdevs
	"""

	#for all positive couplngs we expect this to have the 4 nearly 
	#degenerate eigenvalues greater than the two pairs
	group1, group2, group3 = eigenvalues[:2], eigenvalues[2:4], eigenvalues[4:]

	tot_std = np.std(group1) + np.std(group2) + np.std(group3)

	for _ in range(100):
		np.random.shuffle(eigenvalues)
		test_group1 = eigenvalues[:2]
		test_group2 = eigenvalues[2:4]
		test_group3 = eigenvalues[4:]

		test_tot_std = np.std(test_group1) + np.std(test_group2) + np.std(test_group3)

		assert(test_tot_std >= tot_std)

	return True
TEST_get_projected_energies()


def TEST_identify_levels():
	eigs = [-10, -9, 0, 1, 10, 11, 13, 14]

	l3, pair = identify_levels(eigs)
	l1, l2 = pair

	assert(l1 == -9.5)
	assert(l2 == .5)
	assert(l3 == 12)

	return True
TEST_identify_levels()


def TEST_fit_to_values():
	"""
	confirm that this works in a trivial case
	"""
	errors = []
	for _ in range(50):
		j34, j35, j45 = np.random.rand(3)+.001

		val1 = (1/4)*(-2*np.sqrt(j34**2 - j34*j35 +j35**2 -j45*(j34+j35) +j45**2) -j34-j35-j45)
		val2 = (1/4)*(2*np.sqrt(j34**2 - j34*j35 +j35**2 -j45*(j34+j35) +j45**2) -j34-j35-j45)
		val3 = (j34+j35+j45)/4

		eigenvalues = np.array([val1, val1, val2, val2, val3, val3, val3, val3])

		fitted_j35, fitted_j45 = sorted(fit_to_values(eigenvalues, j34))
		j35, j45 = sorted([j35, j45])

		#print("found: ", fitted_j35, fitted_j45)
		#print("actual ", j35, j45)
		
		errors.append(abs(fitted_j45-j45)/j45)
		errors.append(abs(fitted_j35-j35)/j35)

	#I want mean error less than 3 percent error	
	assert(np.mean(errors)<.03)
	if np.mean(errors)>0.01:
		print("Average Coupling Fitting Error: ", np.mean(errors))
	return True
TEST_fit_to_values()



def TEST_max_coupling():
	"""
	Lets confirm that the maximum coupling is in fact the maximum coupling

	I'll make a trivial graph where each point is connected to 1 other point

	Then set one to be huge, and make sure this finds it
	"""
	points = np.random.rand(100, 2)
	point_dict = dict()

	huge_index = 30

	for i in range(points.shape[0]):
		p = (points[i][0], points[i][1])
		i1 = (i+1)%points.shape[0]
		connected_p = (points[i1][0], points[i1][1])

		if (p not in point_dict) and (connected_p not in point_dict):
			if i!= huge_index:
				strength = np.random.rand()
			else:
				strength = 100

			point_dict[p] = [[connected_p, strength]]
			point_dict[connected_p] = [[p, strength]]

	Jmax, _ = max_coupling(points, point_dict)

	assert(Jmax==100)

	return True
TEST_max_coupling()


def TEST_check_cluster():
	"""
	confirm that this works
	"""

	cluster = (1, 2, 3, 4)
	point_dict = {p1 : [(p2, 1) for p2 in cluster if p2!=p1] for p1 in cluster}
	should_be_true = check_cluster(cluster, point_dict)
	assert(should_be_true)

	cluster = (1, 2, 3)
	point_dict = {p1 : [(p2, 1) for p2 in cluster if p2!=p1] for p1 in cluster}
	point_dict[4] = []
	should_be_false = check_cluster((1,2,3,4), point_dict)
	assert(not should_be_false)

	return True
TEST_check_cluster()



def TEST_get_cluster_couplings():

	J01, J02, J03, J12, J13, J23 = 1, 2, 3, 4, 5, 6

	cluster = (0,1,2,3)
	point_dict = {0 : [(3, J03), (2, J02), (1, J01)],
				1 : [(3, J13), (2, J12), (0, J01)],
				2 : [(3, J23), (1, J12), (0, J02)],
				3 : [(2, J23), (1, J13), (0, J03)]}

	nJ01, nJ02, nJ03, nJ12, nJ13, nJ23 = get_cluster_couplings(cluster, point_dict)

	assert(nJ01 == J01)
	assert(nJ02 == J02)
	assert(nJ03 == J03)
	assert(nJ12 == J12)
	assert(nJ13 == J13)
	assert(nJ23 == J23)

	return True
TEST_get_cluster_couplings()


def TEST_remove_point():
	"""
	apply this to a trivial dictionary
	"""
	J01, J02, J03, J12, J13, J23 = 1, 2, 3, 4, 5, 6
	points = [(0,),(1,),(2,),(3,)]
	point_dict = {(0,) : [((3,), J03), ((2,), J02), ((1,), J01)],
				(1,) : [((3,), J13), ((2,), J12), ((0,), J01)],
				(2,) : [((3,), J23), ((1,), J12), ((0,), J02)],
				(3,) : [((2,), J23), ((1,), J13), ((0,), J03)]}

	#remove 0
	new_points, new_point_dict = remove_point([(0,)], points, point_dict)

	points_confirm = [(1,),(2,),(3,)]
	point_dict_confirm = {(1,) : [((3,), J13), ((2,), J12)],
				(2,) : [((3,), J23), ((1,), J12)],
				(3,) : [((2,), J23), ((1,), J13)]}

	assert(np.sum(np.abs(points_confirm - new_points))==0)

	assert(point_dict_confirm == new_point_dict)

	return True
TEST_remove_point()



def TEST_update_coupling():
	"""
	Apply this to a simple example
	"""
	J01, J02, J03, J12, J13, J23 = 1, 2, 3, 4, 5, 6
	points = [0,1,2,3]
	point_dict = {0 : [(3, J03), (2, J02), (1, J01)],
				1 : [(3, J13), (2, J12), (0, J01)],
				2 : [(3, J23), (1, J12), (0, J02)],
				3 : [(2, J23), (1, J13), (0, J03)]}

	p1, p2 = 0,1
	Jnew = 100
	new_point_dict = update_coupling(p1, p2, Jnew, point_dict)
	
	confirm_point_dict = {0 : [(1, 100), (3, J03), (2, J02)],
				1 : [(0, 100), (3, J13), (2, J12)],
				2 : [(3, J23), (1, J12), (0, J02)],
				3 : [(2, J23), (1, J13), (0, J03)]}

	assert(new_point_dict == confirm_point_dict)

	"""
	now to one where there isnt a connection, but we want to add one
	make sure the ordering of the lists is correct, and that 
	we've added in the right couplings
	"""

	points = [0,1, 3]
	point_dict = {0 : [(3, 50), (2, -1)], 1 : [(2, 1)], 2 : [(1, 1), (0, -1)], 3 :[(0, 50)]}

	p1, p2 = 0,1
	Jnew = 100
	new_point_dict = update_coupling(p1, p2, Jnew, point_dict)
	
	confirm_point_dict = {0 : [(1, 100), (3, 50), (2, -1)], 1 : [(0, 100), (2, 1)], 2 : [(1, 1), (0, -1)], 3 :[(0, 50)]}

	assert(new_point_dict == confirm_point_dict)

	return True
TEST_update_coupling()

print('all tests successful')




