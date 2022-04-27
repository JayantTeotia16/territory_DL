import numpy as np
def fun(d0,d1,d2,att,velocity):
	A = np.array([[3, 6, 7], [5, -3, 0]])
	B = np.array([[1, 1], [2, 1], [3, -3]])


	vec = np.array( [5, 5 ])
	vec_norm = np.linalg.norm(vec) 
	  
	#print("Vector norm:", vec_norm) 

	vec = np.array( [5, 5 ])
	#defender position
	D0 = np.array([10, 10])
	D1 = np.array([ 10 ,-10 ])
	D2 = np.array([-10,10])
	D3 = np.array([-10, -10])
	D0 = d0
	D1 = d1
	D2 = d2
	#estimated arrival pos of intruder
	I0 = np.array([-10, 10])
	I1 = np.array([10, 0])
	I2 = np.array([-10, 40])
	I3 = np.array([-30, 5])
	I4 = np.array([-10, 35])
	I5 = np.array([-25, 50])
	I6 = np.array([20, -30])

	# Input positions of defenders, intruders and velocity of intruuder
	D = np.array([D0,D1, D2  ])
	I = np.array([I0,I1, I2, I3, I4])#,I1, I2, I3, I4, I5, I6 ])
	#print(att)
	I = np.array(att)
	vlimit_intru = np.array(velocity[0])
	#vlimit_intru = np.array([1, 0.12, 0.15, 0.1, 0.08, 0.1, 0.12, 0.1, 0.75])


	# Compute task location and time
	D = D - 5
	I = I - 5
	T_pos = I
	T_time = [ 100,120, 120,120,120]
	Dist = [np.linalg.norm(a) for a in I]
	T_time = [Dist[i]/vlimit_intru[i] for i in range(np.size(Dist))]
	#    Compute cost table

	N = D.shape[0]
	M = I.shape[0]
	L = 1000
	#max vel def
	V_d_max = 3
	#print(N,M)
	cost = np.full((N+M,M), 0)
	for i in range(N):
	    for j in range(M):
	        relative_dist = T_pos[j] - D[i]
	        cc = np.linalg.norm(relative_dist)  
	        if cc/V_d_max <= T_time[j]:
	            cost[i,j] = cc
	        else:    
	            cost[i,j] = L

	#print(N)
	for k in range(M):
	    for j in range(M): 
	        if k < j:
	            relative_dist = T_pos[j] - T_pos[k] 
	            cc = np.linalg.norm(relative_dist)  
	            if cc  <= T_time[j]*V_d_max:
	                cost[k+N,j] = cc
	            else:    
	                cost[k+N,j] = L
	        else:
	            cost[k+N,j] = 10000*L
	#print(cost)    


	# solve assignement problem
	from scipy.optimize import linear_sum_assignment 
	row_ind, col_ind = linear_sum_assignment(cost)
	 
	total = cost[row_ind, col_ind].sum() 

	#print(row_ind,col_ind,total) 

	# Find time infeasible tasks
	# add reserve defender 
	# solve assignemnt problem for updated problem

	# from final assignemt assign the target location for defenders
	Target_pos = D
	#print(row_ind.shape[0])
	Def_x = np.ones(np.size(D,0))
	Def_x = Def_x * 1000
	Def_y = np.ones(np.size(D,0))
	Def_y = Def_y * 1000
	Def_z = np.ones(np.size(D,0))
	Def_z = Def_y * 1000
	for i in range(N):
	#	print("hii")
		#print(["i=",i])	
		ii = col_ind[np.where(row_ind == i)]
		#print(ii)
		if ii.shape[0] > 0:
			j +=1
			#print("i=",i," ii=",ii)	
			Def_x[i] = I[ii, 0]
			Def_y[i] = I[ii, 1]
			Def_z[i] = ii
	return Def_x, Def_y, Def_z
