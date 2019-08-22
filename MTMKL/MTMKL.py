"""Implements Multi-task Multi-kernel Learning (MTMKL)

This multi-task learning (MTL) classifier learns a set of kernels for different 
groups of features (or feature modalities). Each task learns to combine these
kernels with a different set of weights. The weights are regularized globally
to share information among the tasks.

This model was originally proposed in:
Kandemir, M., Vetek, A., Goenen, M., Klami, A., & Kaski, S. (2014). 
Multi-task and multi-view learning of user state. Neurocomputing, 139, 97-106.
"""
import numpy as np
import scipy.optimize as opt
import scipy.linalg as la
from scipy import interp
import math

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances, cosine_similarity
import numpy.linalg as LA

import pandas as pd
import sys
import os
import random
import pickle
import copy
import operator
import datetime

from scipy.optimize import minimize

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

import helperFuncs as helper
from LSSVM import LSSVM

def reloadFiles():
	reload(helper)
	print "Cannot reload LSSVM because of the way it was imported"


reloadFiles()

DEBUG = False
VERBOSE = False

class MTMKL:
	def __init__(self, task_dict_list, C=100.0, V=0.1, kernel_name='rbf', kernel_param=.01, regularizer=None, max_iter=50, 
					max_iter_internal=-1, tol=0.001, eta_filename=None, debug=DEBUG, verbose=VERBOSE, drop20PercentTrainingData=False):
		'''INPUTS:
				task_dict_list: 	a particular format, defined here: https://docs.google.com/document/d/1BlMaluZnPTa0oznWrfy5sku44ydunv_kalGfG_yz49c/edit?usp=sharing'''
		#possible kernels: linear, 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable

		#data features
		self.train_tasks = task_dict_list
		self.val_tasks = None
		self.test_tasks = None

		self.modality_names, self.modality_start_indices = self.getModalityNamesIndices(task_dict_list)
		self.modality_start_indices.append(np.shape(task_dict_list[0]['X'])[1]) #append the number of columns 

		self.n_tasks = len(self.train_tasks)		#number of tasks
		self.n_views = len(self.modality_names)		#number of views (one view can be one sensor of feature set e.g. physiology features)
		self.eta = np.array([[1.0/self.n_views] * self.n_views] * self.n_tasks)		#a matrix of size number of tasks x number of sensors
		self.last_eta = self.eta

		self.eta_filename = eta_filename
		if eta_filename is not None:
			eta_file = open(self.eta_filename,'w')
			#eta_file.write("//Eta matrix")
			eta_file.close()
			self.save_etas = True
		else:
			self.save_etas = False

		#MTMKL parameters
		self.V = V 				#V is a weight placed on the regularization . Small corresponds to unrelated tasks. 
								#Large is enforcing similar kernel weights across tasks
								#Kandemir et al. recommends testing a range from 10^-4 to 10^4
								#V=0 is an independent, multi-kernel learner for each task
		self.C = C 				#C parameter for SVM classifiers
		self.regularizer = regularizer
		self.max_iter = max_iter 					#max iterations that MTMKL algorithm will run for
		self.regularizer_func= None
		self.regularizing_grad = None
		self.kernel_name = kernel_name
		self.setKernel(kernel_name, kernel_param)
		self.setRegularizer(regularizer)

		#internal SVM parameters
		self.max_iter_internal = max_iter_internal 	#max iterations for each scikit learn SVM within MTMKL
		self.tolerance = tol 						#convergence criteria for each scikit learn SVM within MTMKL
		
		self.classifiers = [0] * self.n_tasks

		self.debug=debug
		self.verbose=verbose
		self.drop20 = drop20PercentTrainingData

		if self.debug: print "MTMKL class has been initialized with", self.n_tasks, "tasks and", self.n_views, "sensors"

	@staticmethod
	def getModalityNamesIndices(task_dict_list):
		modality_dict = task_dict_list[0]['ModalityDict']
		sorted_tuples = sorted(modality_dict.items(), key=operator.itemgetter(1))
		names = [n for (n,i) in sorted_tuples] 
		indices = [i for (n,i) in sorted_tuples]
		return names,indices

	def setTrainData(self, task_dict_list):
		self.train_tasks = task_dict_list

	def setTestData(self, task_dict_list):
		self.test_tasks = task_dict_list

	def setValData(self, task_dict_list):
		self.val_tasks = task_dict_list

	def setC(self, c):
		self.C = c

	def setV(self, V):
		self.V = V

	def setKernel(self, kernel_name, kernel_param):
		self.kernel_name = kernel_name
		if kernel_name == 'rbf':
			def rbf(x1,x2):
				return rbf_kernel(x1,x2, gamma=kernel_param) # from sklearn

			self.internal_kernel_func = rbf
		else:
			def dot_product(x1,x2):
				return cosine_similarity(x1,x2) # from sklearn - a normalized version of dot product #np.dot(x1,x2.T)
			self.internal_kernel_func = dot_product

	def setRegularizer(self,regularizer):
		self.regularizer = regularizer
		if regularizer == 'L1':
			self.regularizer_func = self.eta_L1
			self.regularizing_grad = self.eta_grad_L1
		else:
			self.regularizer_func = self.eta_L2
			self.regularizing_grad = self.eta_grad_L2

	def setAllSettings(self, c, v, kernel, beta, regularizer):
		self.setC(c)
		self.setV(v)
		self.setKernel(kernel,beta)
		self.setRegularizer(regularizer)

	#kernel will know which column indices belong to which sensor 
	def constructKernelFunction(self, task):
		task_eta = self.eta[task,:]

		def overallKernel(X1,X2): #change to static
			K = np.zeros((len(X1),len(X2)))

			for m in range(self.n_views):
				sub_x1 = X1[:,self.modality_start_indices[m]:self.modality_start_indices[m+1]]
				sub_x2 = X2[:,self.modality_start_indices[m]:self.modality_start_indices[m+1]]

				internal_K = self.internal_kernel_func(sub_x1,sub_x2)
				
				K = K + task_eta[m] * internal_K/np.max(abs(internal_K))

			return K

		return overallKernel

	def eta_L1(self):
		return -self.V*np.sum(np.dot(self.eta,self.eta.T))

	def eta_L2(self):
		# Note that V should be positive
		return self.V*np.sum(euclidean_distances(self.eta,squared=True))

	def eta_grad_L1(self, eta_mat,v,task_index):
		return -v*np.sum(eta_mat,axis=0)

	def eta_grad_L2(self, eta_mat,v,task_index):
		# Note that V should be positive
		return 2*v*np.sum(eta_mat[task_index,:]-eta_mat,axis=0)

	def computeObjectiveFunction(self,eta_from_fmin):
		eta_from_fmin = eta_from_fmin.reshape(self.n_tasks,-1)
		#if self.debug: print "eta:", eta_from_fmin
		if self.debug: print "sum eta per task:", np.sum(eta_from_fmin,axis=1)
		if self.save_etas:
			self.saveEtas()
		self.eta = eta_from_fmin

		#steps 1 and 2 of Kandemir algorithm
		for t in range(self.n_tasks):
			if self.debug: 
				print "Training task", t
				print "etas have size", self.eta.shape
				sys.stdout.flush()

			X_t, Y_t = self.extractTaskData(self.train_tasks,t,drop20=self.drop20)

			overallKernel = self.constructKernelFunction(t)

			self.classifiers[t] = LSSVM.LSSVM(self.C,kernel_func=overallKernel)
			#SVC(C=self.C, kernel=overallKernel, probability=True, max_iter=self.max_iter_internal, tol=self.tolerance)
			converged = self.classifiers[t].fit(X_t, Y_t)
			assert converged



		# Compute the objective function
		obj_value = 0
		for t in range(self.n_tasks):
			X_t, Y_t = self.extractTaskData(self.train_tasks,t,drop20=self.drop20)

			alpha = self.classifiers[t].alphas

			overallKernel = self.constructKernelFunction(t)
			K = overallKernel(X_t,X_t)
		
			obj_value += sum(alpha)-(0.5*1.0/self.C)*sum(alpha**2) -(1.0/2.0)*(np.dot((alpha*Y_t).T,np.dot(K,alpha*Y_t)))

		# add regularizer
		obj_value += self.regularizer_func()

		if self.debug: 
			print "obj function value:", obj_value
			print "Eta difference:",self.computeEtaDifference()
			print "Training ACC", self.predictAndGetAccuracy(self.train_tasks)
			print 

		return obj_value


	# eta_mat has rows for tasks, columns for sensors
	def computeMatrixGradient(self,eta_from_fmin):
		update = np.zeros((self.n_tasks,self.n_views))
		
		for t in range(self.n_tasks):
			X_t, Y_t = self.extractTaskData(self.train_tasks,t,drop20=self.drop20)

			alpha = self.classifiers[t].alphas
			alphaY = alpha*Y_t

			for m in range(self.n_views): #Used to be numSensors-1
				sub_x1 = X_t[:, self.modality_start_indices[m]:self.modality_start_indices[m+1]]
				sub_x2 = X_t[:, self.modality_start_indices[m]:self.modality_start_indices[m+1]]

				# Normalize the kernel, could also use  k(i, j) = k (i, j) / sqrt(k(i,i) * k(j,j))
				#note, the same procedure for finding the min of sub_x1 and sub_x2 that is used in
				#the overall kernel is not required here, since sub_x1 and sub_x2 are guaranteed
				#to be the same
				internal_K = self.internal_kernel_func(sub_x1,sub_x2)

				update[t,m] = -(1.0/2.0)*(np.dot(alphaY.T,np.dot(internal_K,alphaY)))

			grad_reg = self.regularizing_grad(eta_from_fmin.reshape(self.n_tasks,-1),self.V,t)

			update[t,:] = grad_reg + update[t,:]

		return update.flatten()  

	def saveEtas(self):
		if self.eta_filename is not None:
			eta_file = open(self.eta_filename,'a')
			np.savetxt(eta_file,self.eta.flatten())
			eta_file.close()

	def computeEtaDifference(self):
		max_diff = 0
		for t in range(self.n_tasks):
			last_eta_list = self.last_eta[t,:]
			eta_list = self.eta[t,:]

			norm = la.norm(last_eta_list - eta_list)
			
			if norm > max_diff:
				max_diff = norm
		return max_diff

	def createConstraintList(self):
		constraints = []

		# Equality constraints
		for t in range(self.n_tasks):
			start = t*self.n_views
			end = (t+1)*self.n_views
			def fun_eq(x,start=start, end=end):
				res = np.array([np.sum(x[start:end])-1.0])
				return res
			def jac_func(x,start=start,end=end):
				jac= np.zeros(self.n_tasks*self.n_views)
				jac[start:end] = 1.0
				return jac
			cons = {'type':'eq',
					'fun':fun_eq,
					'jac':jac_func}
			constraints.append(cons)

		# Inequality constraints
		for i in range(self.n_tasks*self.n_views):
			def jac_func(x,i=i):
				jac= np.zeros(self.n_tasks*self.n_views)
				jac[i] = 1.0
				return jac
			cons = {'type':'ineq',
					'fun':lambda x,i=i: np.array([x[i]]),
					'jac':jac_func}
			constraints.append(cons)

		return constraints

	def train(self):
		init_etas = self.eta.flatten()
		cons = self.createConstraintList()
		try:
			res = minimize(self.computeObjectiveFunction, init_etas, jac=self.computeMatrixGradient,constraints=cons, method='SLSQP', options={'disp': self.verbose,'maxiter':self.max_iter})
		except:
			return False
		self.eta = res.x.reshape(self.n_tasks,-1)

		if self.verbose: 
			print "Results of this run!"
			print "\t ETA", self.eta
			print "\t Training ACC", self.predictAndGetAccuracy(self.train_tasks)

		return True


	@staticmethod
	def extractTaskData(task_dict_list,t,drop20=False):
		X_t = task_dict_list[t]['X']
		Y_t = (task_dict_list[t]['Y']).reshape(-1,1)

		if drop20:
			keep_indices = task_dict_list[t]['KeepIndices']
			X_t = X_t[keep_indices]
			Y_t = Y_t[keep_indices]

		return X_t, Y_t

	def predict(self, task_dict_list):
		''' input: 		task_dict_list in the usual format. Will not use the 'Y' key
			output:		predictions for the y values for each task. So a list of lists, where each inner list
						is the y_hat values for a particular task'''
		Y_hat = [0] * len(task_dict_list)
		for t in range(len(task_dict_list)): 
			Y_hat[t] = self.predictOneTask(task_dict_list,t)
		return Y_hat

	def predictOneTask(self, task_dict_list, t):
		X_t, y_t = self.extractTaskData(task_dict_list,t)
		if len(X_t) == 0:
			return None
		else:
			return self.internal_predict(X_t, int(t))

	def internal_predict(self, X_t, t):
		return self.classifiers[t].predict(X_t).reshape(-1,1)

	def predict_01(self, X, t):
		preds = self.classifiers[t].predict(X).reshape(-1,1)
		return (preds + 1.0) / 2

	def getNumErrors(self, Y, Y_hat):
		#returns accuracy
		errors = np.where(Y * Y_hat < 0)[0] 
		return len(errors)

	def getAccuracy(self, Y, Y_hat):
		score = self.getNumErrors(Y,Y_hat)
		return 1.0 - (float(score) / float(len(Y_hat)))

	def predictAndGetNumErrors(self,task_dict_list):
		Y_hat = self.predict(task_dict_list)
		return self.getNumErrors(task_dict_list['Y'],Y_hat)

	def predictAndGetAccuracy(self,task_dict_list):
		Y_hat = self.predict(task_dict_list)
		accs = []#[0] * len(task_dict_list)
		for t in range(len(task_dict_list)):
			accs.append(self.getAccuracy(task_dict_list[t]['Y'],Y_hat[t]))
		return np.mean(accs)

	def predictAndGetAccuracyOneTask(self,task_dict_list,t):
		Y_hat = self.predictOneTask(task_dict_list,t)
		return self.getAccuracy(task_dict_list[t]['Y'],Y_hat[t])

	def getAccuracyAucAllTasks(self, tasks):
		all_task_Y = []
		all_preds = []
		for t in range(len(tasks)):
			X_t, y_t = self.extractTaskData(tasks,t)
			if len(X_t) == 0:
				continue
			preds = self.internal_predict(X_t, int(t))
			all_task_Y.extend(y_t)
			all_preds.extend(preds)
		auc=roc_auc_score(all_task_Y, all_preds)
		acc=helper.getBinaryAccuracy(all_preds,all_task_Y)
		return acc,auc

	def getAccuracyAucOnOneTask(self, task_list, task, debug=False):
		X_t, y_t = self.extractTaskData(task_list,task)
		if len(X_t) == 0:
			return np.nan, np.nan
		
		preds = self.internal_predict(X_t, int(task))

		if debug:
			print "y_t:", y_t
			print "preds:", preds
		
		acc = helper.getBinaryAccuracy(preds,y_t)
		if len(y_t) > 1 and helper.containsEachSVMLabelType(y_t) and helper.containsEachSVMLabelType(preds):
			auc = roc_auc_score(y_t, preds)
		else:
			auc = np.nan

		return acc, auc

	def getAUC(self,test_tasks):
		mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		for t in range(self.n_tasks):
			X_t, Y_t = self.extractTaskData(self.train_tasks,t)
			X_test_t, Y_test_t = self.extractTaskData(test_tasks, t)

			overallKernel = self.constructKernelFunction(t)

			self.classifiers[t] = SVC(C=self.C, kernel=overallKernel, probability=True, max_iter=self.max_iter_internal, tol=self.tolerance)
			probas_ = self.classifiers[t].fit(X_t, Y_t).predict_proba(X_test_t)
			fpr, tpr, thresholds = roc_curve(Y_test_t, probas_[:, 1])

			mean_tpr += interp(mean_fpr, fpr, tpr)
			mean_tpr[0] = 0.0

		mean_tpr /= self.n_tasks
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)

		return mean_auc, mean_fpr, mean_tpr

	def getAUCOneTask(self,test_tasks,t):
		global eta_global

		X_t, Y_t = self.extractTaskData(self.train_tasks,t)
		X_test_t, Y_test_t = self.extractTaskData(test_tasks, t)

		overallKernel = self.constructKernelFunction(t)

		self.classifiers[t] = SVC(C=self.C, kernel=overallKernel, probability=True, max_iter=self.max_iter_internal, tol=self.tolerance)
		probas_ = self.classifiers[t].fit(X_t, Y_t).predict_proba(X_test_t)
		fpr, tpr, thresholds = roc_curve(Y_test_t, probas_[:, 1])

		return auc(fpr, tpr), fpr, tpr

	def saveClassifierToFile(self, filepath):
		s = pickle.dumps(self.classifier)
		f = open(filepath, 'w')
		f.write(s)

	def loadClassifierFromFile(self, filepath):
		f2 = open(filepath, 'r')
		s2 = f2.read()
		self.classifier = pickle.loads(s2)




