"""Performs hyperparameter sweep for the logistic regression (LR) model."""
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
import LR as lr

#Parameter values
C_VALS = [ 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] 
PENALTIES = ['l1', 'l2']
SOLVER = 'liblinear' 				#newton-cg, lbfgs, liblinear, sag
DEFAULT_VALIDATION_TYPE = 'cross' 	#'cross' for cross-validation, 'val' for single validation
DEFAULT_NUM_CROSS_FOLDS = 5
NUM_BOOTSTRAPS = 5
VERBOSE = True						#set to true to see more output
SAVE_RESULTS_EVERY_X_TESTS = 1

def reload_dependencies():
	reload(helper)
	reload(lssvm)

class LRWrapper(STLWrapper):
	def __init__(self, file_prefix, users_as_tasks=False, cont=False, c_vals=C_VALS, 
				 penalties=PENALTIES, solver=SOLVER, num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, 
				 dropbox_path=PATH_TO_DROPBOX, datasets_path='Data/',
				 test_csv_filename=None):
		self.c_vals = c_vals
		self.penalties = penalties
		self.solver = solver
		
		STLWrapper.__init__(self, file_prefix, users_as_tasks=users_as_tasks, cont=cont, 
				classifier_name='LR', num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
				datasets_path=datasets_path, cant_train_with_one_class=False, 
				save_results_every_nth=SAVE_RESULTS_EVERY_X_TESTS, test_csv_filename=test_csv_filename)

		self.models = [None] * self.n_tasks

	def define_params(self):
		self.params = {}
		self.params['C'] = self.c_vals
		self.params['penalty'] = self.penalties
	
	def train_and_predict_task(self, t, train_X, train_y, eval_X, param_dict):
		self.models[t] = lr.LR(penalty=param_dict['penalty'], C=param_dict['C'], solver=self.solver)
		self.models[t].setTrainData(train_X, train_y)
		self.models[t].train()
		preds = self.models[t].predict(eval_X)

		return preds 

	def predict_task(self, X, t):
		if self.models[t] is None:
			print "ERROR! No model has been trained!"
			
		preds = self.models[t].predict(X)
		return (preds + 1.0) / 2

	# use something like the following to test only one set of parameters:
	# wrapper.setParams(c_vals=[10], beta_vals=[.01], kernels=['rbf'])
	def set_params(self, c_vals=None, penalties=None, solver=None):
		'''does not override existing parameter settings if the parameter is not set'''
		self.c_vals = c_vals if c_vals is not None else self.c_vals
		self.penalties = penalties if penalties is not None else self.penalties
		self.solver = solver if solver is not None else self.solver
		self.define_params()

	
if __name__ == "__main__":
	print "LOGISTIC REGRESSION (LR) MODEL SELECTION"
	print "\tThis code will sweep a set of parameters to find the ideal settings for LR for a single dataset"

	if len(sys.argv) < 3:
		print "Error: usage is python LRWrapper.py <file prefix> <users as tasks> <continue>"
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

	wrapper = LRWrapper(file_prefix, users_as_tasks=users_as_tasks, cont=cont,
						test_csv_filename=csv_test_file)
	
	print "\nThe following parameter settings will be tested:"
	print "\tC_VALS:  	\t", wrapper.c_vals
	print "\tPENALTIES:   	\t", wrapper.penalties
	
	print "\nOptimization will be performed with the following solver:"
	print "\tSolver:   \t", wrapper.solver

	print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'
	print "\nThe validation and testing figures will be saved in:", wrapper.figures_path + wrapper.save_prefix

	wrapper.run()

