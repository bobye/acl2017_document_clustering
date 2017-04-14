"""
Compute Categorical Cluster Distance
"""

#Author: Yukun Chen
#Contact; cykustc@gmail.com
#Copyright: Penn State University, 

import csv
import numpy as np
import sys
from scipy.spatial import distance
from cvxopt import matrix, solvers

def read_tokens(cluster_result_file):
	"""read clustering result from csv file (only one record each row) to a list"""
	with open(cluster_result_file,'rb') as csvfile:
		filereader=csv.reader(csvfile,delimiter=',');
		result=list();
		for row in filereader:
			result.append(row[0]);
	return result

def cluster_info(clustering_result):
	"""return cluster number: clust_num and cluster name dictionary: cluster_dict in a tuple (clust_num,cluster_dict)
	"""
	cluster_set=list(set(clustering_result));
	cluster_num=len(cluster_set);
	cluster_dict=dict(zip(cluster_set,range(cluster_num)));
	return (cluster_num,cluster_dict)

def token_to_mat(tokens):
	cluster_num, cluster_dict = cluster_info(tokens);
	N = len(tokens);
	clus_mat = np.zeros((N,cluster_num));
	for i in xrange(N):
		clus_mat[i][cluster_dict[tokens[i]]]=1;
	return clus_mat

def categorical_clust_dist(clus_mat_A,clus_mat_B,method='even'):
	"""
	Compute Clustering distance from [clusters clus_mat_A with weights w_A] to [clusters clus_mat_B with weights w_B]
	
	More details please refer to Section 4.1 of "A New Mallows Distance Based
	Metric for Comparing Clusterings", Ding Zhou, Jia Li, Hongyuan Zha.

	Return a dictionary contains the Categorical Cluster Distance and matching weights {"dist":,"matching"}
	"""
	n = clus_mat_A.shape[1];
	m = clus_mat_B.shape[1];
	if method=='even':
		w_A=1.0/n*np.ones(n)
		w_B=1.0/m*np.ones(m)
	elif method=='instance_count':
		w_A=np.sum(clus_mat_A,axis=0)
		w_A=w_A/np.sum(w_A)
		w_B=np.sum(clus_mat_B,axis=0)
		w_B=w_B/np.sum(w_B)
	A = np.zeros((n+m,n*m));
	for k in xrange(n):
		A[k][np.arange(k,n*m,n)]=1;
	for k in xrange(m):
		A[n+k][np.arange(k*n,k*n+n)]=1;
	A = A[:-1,:]
	D = distance.cdist(clus_mat_A.T,clus_mat_B.T,'cityblock'); #Computes the city block or Manhattan distance 
	f = D.reshape((1,n*m));
	b = np.concatenate((w_A.T,w_B.T),axis=0)
	b = b[:-1]
	c = matrix(f.T);
	beq = matrix(b);
	Aeq = matrix(A);

	G = matrix(-1.0*np.eye(m*n),(m*n,m*n))
	h = matrix(0,(m*n,1),'d')

	solvers.options['show_progress'] = False
	sol = solvers.lp(c,G,h,A=Aeq,b=beq);
	x=sol['x'];
	#print sol
	x=np.array(x);
	x=x.reshape((n,m), order='F')
	return {"dist":sol['primal objective'],"matching":x}


if __name__ == '__main__':
	clus_mat_A = token_to_mat(read_tokens('cluster_resultsA.txt'))
	clus_mat_B = token_to_mat(read_tokens('cluster_resultsB.txt'))
	# print clus_mat_A, clus_mat_B
	if clus_mat_A.shape[0]!=clus_mat_B.shape[0]:
		print "number of instances in two clustering result are not the same!";
		sys.exit(0)
	result=categorical_clust_dist(clus_mat_A,clus_mat_B,method='instance_count')
	# result=categorical_clust_dist(clus_mat_A,clus_mat_A,method='even')
	print result['dist']
	print result['matching']




