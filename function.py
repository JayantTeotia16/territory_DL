import numpy as np
from scipy.optimize import linear_sum_assignment 
def fun(defender,att,time):
	# Input positions of defenders, intruders and velocity of intruuder
    D = np.array(defender)
    I = np.array(att)
    T_time = np.array(time)
    D = D - 5
    I = I - 5
    T_pos = I
    #    Compute cost table

    N = D.shape[0]
    M = I.shape[0]
    L = 1000
    #max vel def
    V_d_max = 2
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
	    
    row_ind, col_ind = linear_sum_assignment(cost)
    arr =np.zeros((M,2))
    for i in range(M):
        arr[i][0] = row_ind[i] -3
        arr[i][1] = col_ind[i]
    a = []
    for i in range(N):
        a.append([-N+i])
    for i in range(M):
        for j in range(N):	        
            if arr[i][0] == a[j][len(a[j])-1]:
                a[j].append(arr[i][1])
                break
    for j in range(N):
        a[j].pop(0)
    return a
