import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import sys
import os
import copy
from time import time
from sklearn.metrics.pairwise import rbf_kernel

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_RESULTS_PATH = '/Your/path/here/'
DEFAULT_DATASETS_PATH = '/Your/path/here/'
DEFAULT_FIGURES_PATH = '/Your/path/here/'

from generic_wrapper import STLWrapper
import helperFuncs as helper
import LSSVM as lssvm

C_VALS = [0.1, 1.0, 10.0, 100.0] 						#values for the C parameter of SVM to test
BETA_VALS = [.0001, .01, .1, 1]							#values for the Beta parameter of rbf kernel to test
KERNELS = ['linear', 'rbf'] 							#could also try 'poly' and 'sigmoid'
DEFAULT_VALIDATION_TYPE = 'cross' #'val' 				#'cross' for cross-validation, 'val' for single validation
VERBOSE = True											#set to true to see more output
NUM_BOOTSTRAPS = 5
DEFAULT_NUM_CROSS_FOLDS = 5
SAVE_RESULTS_EVERY_X_TESTS = 1

def reload_dependencies():
	reload(helper)
	reload(lssvm)

class LSSVMWrapper(STLWrapper):
	def __init__(self, file_prefix, users_as_tasks=False, cont=False, c_vals=C_VALS, beta_vals=BETA_VALS, 
				 kernels=KERNELS, num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, dropbox_path=PATH_TO_DROPBOX, 
				 datasets_path='Data/', test_csv_filename=None):
		self.c_vals = c_vals
		self.beta_vals=beta_vals
		self.kernels = kernels
		
		STLWrapper.__init__(self, file_prefix, users_as_tasks=users_as_tasks, cont=cont, 
				classifier_name='LSSVM', num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
				datasets_path=datasets_path, cant_train_with_one_class=True, 
				save_results_every_nth=SAVE_RESULTS_EVERY_X_TESTS, test_csv_filename=test_csv_filename)

		self.trim_extra_linear_params()

		self.models = [None] * self.n_tasks

	def define_params(self):
		self.params = {}
		self.params['C'] = self.c_vals
		self.params['beta'] = self.beta_vals
		self.params['kernel'] = self.kernels
	
	def train_and_predict_task(self, t, train_X, train_y, eval_X, param_dict):
		kernel_func = self.get_kernel_func(param_dict['kernel'], param_dict['beta'])
		self.models[t] = lssvm.LSSVM(C=param_dict['C'], kernel_func=kernel_func) 
		converged = self.models[t].fit(train_X,train_y)

		if converged:
			preds = self.models[t].predict(eval_X)
		else:
			# predict majority class
			preds = np.sign(np.mean(train_y))*np.ones(len(eval_X))
		
		return preds 
	
	def predict_task(self, X, t):
		if self.models[t] is None:
			print "ERROR! No model has been trained!"
			
		preds = self.models[t].predict(X)
		return (preds + 1.0) / 2

	# use something like the following to test only one set of parameters:
	# wrapper.setParams(c_vals=[10], beta_vals=[.01], kernels=['rbf'])
	def set_params(self, c_vals=None, beta_vals=None, kernels=None):
		'''does not override existing parameter settings if the parameter is not set'''
		self.c_vals = c_vals if c_vals is not None else self.c_vals
		self.beta_vals = beta_vals if beta_vals is not None else self.beta_vals
		self.kernels= kernels if kernels is not None else self.kernels
		self.define_params()

	def get_kernel_func(self,kernel_name, beta):
		if kernel_name == 'rbf':
			def rbf(x1,x2):
				return rbf_kernel(x1,x2, gamma=beta) # from sklearn
			return rbf
		else:
			def dot_product(x1,x2):
				return np.dot(x1,x2.T)
			return dot_product
	
	def trim_extra_linear_params(self):
		single_beta = None
		i = 0
		while i < len(self.list_of_param_settings):
			setting = self.list_of_param_settings[i]
			if setting['kernel'] == 'linear':
				if single_beta is None:
					single_beta = setting['beta']
				elif setting['beta'] != single_beta:
					self.list_of_param_settings.remove(setting)
					continue
			i += 1
	
if __name__ == "__main__":
	print "LSSVM MODEL SELECTION"
	print "\tThis code will sweep a set of parameters to find the ideal settings for LS SVM for a single dataset"

	if len(sys.argv) < 3:
		print "Error: usage is python LSSVMWrapper.py <file prefix> <users as tasks> <continue>"
		print "\t<file prefix>: e.g. datasetTaskList-Discard-Future-Group_ - program will look in the following directory for this file", DEFAULT_DATASETS_PATH
		print "\t<users as tasks>: type 'users' for users as tasks, or 'wellbeing' for wellbeing measures as tasks"
		print "\t<continue>: optional. If 'True', the wrapper will pick up from where it left off by loading a previous validation results file"
		print "\t<csv file for testing>: optional. If you want to get the final test results, provide the name of a csv file to test on"
		sys.exit()
	file_prefix= sys.argv[1] #get data file from command line argument
	print "\nLoading dataset", DEFAULT_DATASETS_PATH + file_prefix
	print ""

	if sys.argv[2] == 'users':
		users_as_tasks = True
		print "Okay, treating users as tasks. Will not print per-task results"
	else:
		users_as_tasks = False
		print "Okay, treating wellbeing measures as tasks. Will save and print per-task results"

	if len(sys.argv) >= 4 and sys.argv[3] == 'True':
		cont = True
		print "Okay, will continue from a previously saved validation results file for this problem"
	else:
		cont = False
	print ""

	if len(sys.argv) >= 5:
		csv_test_file = sys.argv[4]
		print "Okay, will get final test results on file", csv_test_file
		print ""
	else:
		csv_test_file = None

	wrapper = LSSVMWrapper(file_prefix, users_as_tasks=users_as_tasks, cont=cont, 
						   test_csv_filename=csv_test_file)
	
	print "\nThe following parameter settings will be tested:"
	print "\tC_VALS:  	\t", wrapper.c_vals
	print "\tBETAS:   	\t", wrapper.beta_vals
	print "\tKERNELS:   \t", wrapper.kernels

	print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'
	print "\nThe validation and testing figures will be saved in:", wrapper.figures_path + wrapper.save_prefix

	wrapper.run()

