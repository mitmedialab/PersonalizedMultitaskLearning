''' HIERARCHICAL BAYES LOGISTIC REGRESSION 
'''

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import math
import scipy #scipy.special.psi is the derivative of the log of the gamma function
import scipy.linalg as la
import scipy.special
import copy
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import HBLR_Distribution


ACC_LOGGED_EVERY_N_STEPS = 10

def plotConvergence(metric, title, save_path=None):
    plt.figure()
    plt.plot(metric,'o-')
    plt.xlabel('Iteration')
    plt.ylabel(title)
    if save_path is not None:
    	plt.savefig(save_path)
    	plt.close()
    else:
    	plt.show()

'''Given a dataset, trains the model'''
class HBLR:

	''' DATA FORMAT: a list of dicts. Each list item is a task, indexed by its number. Each task is a dict,
			containing keys 'X' and 'Y', which are the data matrix and label vector, respectively  
			Note that the X matrix should not contain columns like user_id, timestamp
			Each X is of size num points for that task by number features
			Each Y is of size num points for that task by 1 (column vector)

		Conventions:
		-compute functions are for computing internal parameters used in update functions
		-update functions are for computing parameters of the model used for prediction
		'''
	def __init__(self, task_dict, mu=None, sigma=None, tau10=5e-2, tau20=5e-2, K=None, max_iterations=150, 
				xi_tolerance=1e-2,debug=False,verbose=True):
		self.n_tasks = len(task_dict)
		self.K = self.n_tasks if K is None else K

		self.debug = debug
		self.verbose = verbose
		self.task_dict = task_dict
		self.num_feats = np.shape(task_dict[0]['X'])[1]
		# TODO: Should we be checking if every task has the same number of features?

		#hyperparameters
		self.mu = mu if mu is not None else np.zeros((1,self.num_feats)) 
		self.sigma = sigma if sigma is not None else np.eye(self.num_feats) * 10.0 
		self.tau10 = tau10
		self.tau20 = tau20

		#model parameters
		self.phi = None		# np array shape is n_tasks x K
		self.xi = None		# a list of lists. First index is task number, second is data point within task
		self.theta = None	# a matrix of size K x num_feats
		self.gamma = None	# a list of size K of covariance matrices of size num_feats x num_feats
		
		#internal parameters
		self.small_phi1 = None		# a vector of size K-1 used in computing phi
		self.small_phi2 = None		# a vector of size K-1 used in computing phi
		self.s = None				# a matrix of size n_tasks x K using in computing phi
		self.tau1 = None			# used to compute small phi2
		self.tau2 = None			# used to compute small phi2
		self.task_vectors = None	# used to compute theta, matrix of size n_tasks x num_feats, 
									# only computed once at the beginning

		#store metrics for convergence of parameters
		self.xi_convergence_list = []			#take max of abs(prev - new) over all tasks
		self.phi_convergence_list = []			#take norm of prev matrix - new matrix
		self.s_convergence_list = []			#take norm of prev matrix - new matrix
		self.gamma_convergence_list = []		#take max of abs(prev - new) over all clusters
		self.theta_convergence_list = []		#take norm of prev matrix - new matrix

		#
		self.max_iterations = max_iterations
		self.xi_tolerance = xi_tolerance

	def setHyperParameters(self, mu, sigma, tau10, tau20):
		self.mu = mu
		self.sigma = sigma
		self.tau10 = tau10
		self.tau20 = tau20

	def initializeAllParameters(self):
		self.phi = (1.0 / self.K) * np.ones((self.n_tasks,self.K))
		self.theta = np.tile(self.mu, (self.K, 1)) 
		self.gamma = [self.sigma for i in range(self.K)]
		self.xi = [[0]* len(self.task_dict[i]['Y']) for i in range(self.n_tasks)]
		self.computeXi()
		self.tau1 = self.tau10
		self.tau2 = self.tau20
		self.computeSmallPhis()	
		self.computeTaus()
		self.s = np.zeros((self.n_tasks,self.K))
		self.computeTaskVectors()

		self.xi_convergence_list = []
		self.phi_convergence_list = []
		self.s_convergence_list = []
		self.gamma_convergence_list = []
		self.theta_convergence_list = []

		if self.debug:
			print "initial phi", self.phi
			print "initial small phi1", self.small_phi1
			print "initial small phi2", self.small_phi2
			print "initial tau1", self.tau1, "tau2", self.tau2 

	def trainUntilConverged(self):
		self.initializeAllParameters()

		i=0
		while i<self.max_iterations and (i<2 or self.xi_convergence_list[-1]>self.xi_tolerance):
			if self.debug:
				print "----------------"
				print "iteration",i

				plt.imshow(self.phi)
				plt.show()

			prev_xi = copy.deepcopy(self.xi)
			prev_phi = copy.deepcopy(self.phi)
			prev_s = copy.deepcopy(self.s)
			prev_gamma = copy.deepcopy(self.gamma)
			prev_theta = copy.deepcopy(self.theta)

			self.updateAllParameters()
			if self.K>2:
				restart = self.pruneK()
				if restart:
					if self.verbose: print "Restarting now with K=",self.K
					self.initializeAllParameters()
					self.updateAllParameters()
					i=0
					continue
			
			if i % ACC_LOGGED_EVERY_N_STEPS == 0:
				acc = []
				auc = []
				for j in range(len(self.task_dict)):
					preds0 = self.predictBinary(self.task_dict[j]['X'],j)
					task_Y = self.task_dict[j]['Y']
					if 0 in task_Y and 1 in task_Y:
						auc.append(roc_auc_score(task_Y, preds0))
						acc.append(getBinaryAccuracy(preds0,task_Y))
					#else:
					#	print "doesn't have both tasks",j,task_Y
				if self.verbose:
					print "Training. Iteration", i
					if i>0:
						print "\tXi convergence", self.xi_convergence_list[-1]
					print "\tavg training accuracy",np.mean(acc)
					print "\tavg ROC AUC", np.mean(auc),"\n"

			#compute convergence metrics
			if i > 0:
				self.xi_convergence_list.append(computeMatrixConvergence(flattenListLists(prev_xi), flattenListLists(self.xi)))
				self.phi_convergence_list.append(computeMatrixConvergence(prev_phi, self.phi))
				self.s_convergence_list.append(computeMatrixConvergence(0,self.s))
				self.gamma_convergence_list.append(computeListOfListsConvergence(prev_gamma, self.gamma))
				self.theta_convergence_list.append(computeMatrixConvergence(0, self.theta))
				if self.debug: print "Training. Iteration", i, "- Xi convergence:", self.xi_convergence_list[-1]

			i+=1

			sys.stdout.flush()



	def updateAllParameters(self):
		self.computeSMatrix()
		self.updatePhi()
		
		self.computeSmallPhis()
		self.computeTaus()
		self.updateGamma()
		self.updateTheta()
		self.computeXi()

	def computeTaskVectors(self):
		self.task_vectors = np.zeros((self.n_tasks, self.num_feats))
		for m in range(self.n_tasks):
			task_X = self.task_dict[m]['X']
			task_Y = self.task_dict[m]['Y']
			# Note that transposes are different because we are using different notation than in the paper - specifically we use row vectors where they are using column vectors
			self.task_vectors[m,:] = np.dot((task_Y-0.5).T,task_X)

	def pruneK(self):
		num_tasks_in_cluster = self.n_tasks - np.sum(1*(self.phi<1e-16),axis=0)
		for k in range(len(num_tasks_in_cluster))[::-1]:
			if num_tasks_in_cluster[k]==0:
				self.K = self.K - 1
				return True
		return False

	def computeSMatrix(self):
		for m in range(self.n_tasks):
			task_X = self.task_dict[m]['X']
			task_Y = self.task_dict[m]['Y']
			task_xi = np.array(self.xi[m])

			for k in range(self.K):
				# Note that transposes are different because we are using different notation than in the paper - specifically we use row vectors where they are using column vectors

				# This does all data points (n) at once 
				inner = np.dot(np.atleast_2d(self.theta[k,:]).T, np.atleast_2d(self.theta[k,:])) + self.gamma[k]
				diag_entries = np.einsum('ij,ij->i', np.dot(task_X, inner), task_X)
				s_sum = -rhoFunction(task_xi)*diag_entries
				
				s_sum += ((task_Y.T - 0.5)* np.dot(np.atleast_2d(self.theta[k,:]), task_X.T))[0,:]
				s_sum += np.log(sigmoid(task_xi))
				s_sum += (-0.5)*task_xi
				s_sum += rhoFunction(task_xi)*(task_xi**2)
				
				s_sum = np.sum(s_sum)
					
				if k < self.K-1:
					s_sum = s_sum + scipy.special.psi(self.small_phi1[k]) \
									- scipy.special.psi(self.small_phi1[k] + self.small_phi2[k])
				if k > 0:
					for i in range(k):
						s_sum = s_sum + scipy.special.psi(self.small_phi2[i]) \
									- scipy.special.psi(self.small_phi1[i] + self.small_phi2[i])

				
				self.s[m,k] = s_sum
		if self.debug: print "s:", self.s


	def updatePhi(self):
		a = np.array([np.max(self.s, axis=1)]).T #as used in logsumexp trick https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
		self.phi = np.exp(self.s - (a + np.log(np.atleast_2d(np.sum(np.exp(self.s - a),axis=1)).T)))
		if self.debug: 
			print "phi:", self.phi
			
	def computeSmallPhis(self):
		self.small_phi1 = (1 + np.sum(self.phi,axis=0))[0:-1]
		self.small_phi2 = self.tau1 / self.tau2 + np.array([np.sum(self.phi[:,i:]) for i in range(1,self.K)])
		if self.debug: 
			print "small phi1", self.small_phi1
			print "small phi2", self.small_phi2

	def computeTaus(self):
		self.tau1 = self.tau10 + self.K - 1
		tau2_sum = 0
		for k in range(self.K-1):
			tau2_sum = tau2_sum + (scipy.special.psi(self.small_phi2[k]) \
									- scipy.special.psi(self.small_phi1[k] + self.small_phi2[k]))
		self.tau2 = self.tau20 - tau2_sum
		if self.debug: print "tau1", self.tau1, "tau2", self.tau2

	def updateGamma(self):
		task_matrices = np.zeros((self.n_tasks, self.num_feats, self.num_feats))
		for m in range(self.n_tasks):
			rho_vector = rhoFunction(np.array(self.xi[m]))
			rho_vector = rho_vector.reshape((1,-1))				# Make rho vector 2D
			task_X = self.task_dict[m]['X']
			# Note that the transposing doesn't exactly match the paper because our data format is slightly different
			rho_matrix = abs(rho_vector) * task_X.T
			task_matrices[m,:,:] = np.dot(rho_matrix, task_X)  

		for k in range(self.K):
			inner_sum = np.zeros((self.num_feats,self.num_feats))
			for m in range(self.n_tasks):
				inner_sum = inner_sum + self.phi[m,k] * task_matrices[m,:,:]
			self.gamma[k] = la.inv(la.inv(self.sigma) + 2*inner_sum)
			if self.debug: 
				print "gamma computation {0}".format(k), la.det(la.inv(self.sigma) + 2*inner_sum)

	def updateTheta(self):
		for k in range(self.K):
			inner_sum = np.zeros((1,self.num_feats))
			for m in range(self.n_tasks):
				inner_sum = inner_sum + self.phi[m,k] * np.atleast_2d(self.task_vectors[m,:])
			self.theta[k,:] = (np.dot(self.gamma[k],(np.dot(la.inv(self.sigma),self.mu.T) + inner_sum.T)  )).T

	def computeXi(self):
		for m in range(self.n_tasks):
			task_X = self.task_dict[m]['X']
			for n in range(len(task_X)):
				inner_sum = 0
				for k in range(self.K):
					# Note that transposes are different because we are using different notation than in the paper - specifically we use row vectors where they are using column vectors
					inner_sum += self.phi[m,k]*np.dot((np.dot(np.atleast_2d(task_X[n,:]), 
														(np.dot(np.atleast_2d(self.theta[k,:]).T, np.atleast_2d(self.theta[k,:])) + self.gamma[k]))),
														np.atleast_2d(task_X[n,:]).T)
				assert inner_sum >= 0			# This number can't be negative since we are taking the square root

				self.xi[m][n] = np.sqrt(inner_sum[0,0])
				if self.xi[m][n]==0:
					print m,n

	def predictBinary(self, X, task):
		preds = self.predictProbability(task,X) 
		return [1.0 if p>= 0.5 else 0.0 for p in preds.flatten() ]

	def predictProbability(self, task, X):
		prob = 0
		for k in range(self.K):
			numerator = np.dot(np.atleast_2d(self.theta[k,:]),X.T)
			diag_entries = np.einsum('ij,ij->i', np.dot(X, self.gamma[k]), X) ##
			denom = np.sqrt(1.0 + np.pi/8 * diag_entries)
			prob = prob + self.phi[task,k] * sigmoid(numerator / denom)
		return prob


	# Code for Predicting for a new task
	def metropolisHastingsAlgorithm(self, new_task_X, new_task_y,N_sam=1000):
		gauss_weight = (self.tau1/self.tau2)/(self.n_tasks+(self.tau1/self.tau2))
		point_dist_weight = 1.0/(self.n_tasks+(self.tau1/self.tau2))
		point_centers_matrix  = self.theta
		point_weights = [sum([phi_m[k] for phi_m in self.phi]) for k in range(len(self.phi[0]))]
		mu_mult = self.mu[0] # Mu is assumed to be the same for each weight
		sigma_mult = self.sigma[0,0] # Sigma is assumed to be a scalar times the idenity matrix

		dist = HBLR_Distribution.MainDistribution(gauss_weight,point_dist_weight,point_centers_matrix, point_weights,mu_mult,sigma_mult)

		w_dot_array = [np.atleast_2d(dist.rvs(size=1))]
		for i in range(N_sam-1):
			w_hat = np.atleast_2d(dist.rvs(size=1))
			accept_prob = min(1,self.dataProb(new_task_X,new_task_y,w_hat)/self.dataProb(new_task_X,new_task_y,w_dot_array[-1]))
			if np.random.uniform()<accept_prob:
				w_dot_array.append(w_hat)
			else:
				w_dot_array.append(w_dot_array[-1])
		return w_dot_array

	def dataProb(self,new_task_X,new_task_y,weights):
		prod = 1
		for i in range(len(new_task_X)):
			sig = sigmoid(np.dot(weights,np.atleast_2d(new_task_X[i,:]).T ))
			prod = prod*(sig**new_task_y[i]) * (1.0-sig)**(1-new_task_y[i])
	
		return prod

	def predictNewTask(self,new_task_X,new_task_y,pred_X,N_sam=1000):
		w_dot_array = self.metropolisHastingsAlgorithm(new_task_X,new_task_y,N_sam)
	
		predictions = []
		for x_star in pred_X:
			predictions.append(sum([sigmoid(np.dot(w,np.atleast_2d(x_star).T))[0,0] for w in w_dot_array])/float(N_sam))
		predictions = [1.0 if p>=0.5 else 0.0 for p in predictions]
		return predictions


# Helper function
def flattenListLists(listLists):
	return np.array([item for sublist in listLists for item in sublist])

# mathematical helper functions
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def rhoFunction(x):
	assert len(np.where(x==0)[0]) == 0 		#there should not be any zeros passed to this function

	return (0.5 - sigmoid(x)) / (2.0*x)

def computeMatrixConvergence(prev, new):
	return la.norm(new-prev)

def computeListOfListsConvergence(prev, new):
	assert len(prev) == len(new)

	max_diff = 0
	for i in range(len(prev)):
		diff = la.norm(np.array(new[i])-np.array(prev[i]))
		if diff > max_diff:
			max_diff = diff
	return max_diff

def getBinaryAccuracy(pred,true_labels):
	assert len(pred)==len(true_labels)

	correct_labels = [1 for i in range(len(pred)) if pred[i]==true_labels[i]]

	return len(correct_labels)/float(len(pred))
