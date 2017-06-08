import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
#from cvxopt import matrix, solvers
import scipy.linalg as la
import math
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import rbf_kernel
from scipy import interp
import pandas as pd
import sys
import os
import random
import pickle
import copy
import operator
import datetime
from time import time

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_RESULTS_PATH = '/Your/path/here/'
DEFAULT_DATASETS_PATH = '/Your/path/here/'
DEFAULT_FIGURES_PATH = '/Your/path/here/'
DEFAULT_ETAS_PATH = DEFAULT_RESULTS_PATH + 'etas/'

import helperFuncs as helper
import MTMKL as mtmkl

USE_TENSORFLOW = False

C_VALS = [1.0, 10.0, 100.0]   #10.0,100.0, #values for the C parameter of SVM to test
B_VALS = [0.0001, 0.001, 0.01]
V_VALS = [100.0, 10.0, 1.0, .1, .01]               #a small V works well for MKL
REGULARIZERS = ['L1','L2']       
KERNELS = ['rbf','linear'] # could also do linear

VALIDATION_TYPE = 'cross'
DEFAULT_NUM_CROSS_FOLDS = 5
SAVE_RESULTS_EVERY_X_TESTS = 1


def reloadFiles():
	reload(helper)
	reload(mtmkl)
	mtmkl.reloadFiles()


class MTMKLWrapper:
	def __init__(self, file_prefix, users_as_tasks, user_clusters=True, eta_filename=None, regularizers=REGULARIZERS, tolerance = .0001, 
				max_iter = 100, val_type=VALIDATION_TYPE, c_vals=C_VALS, beta_vals=B_VALS, 
				v_vals = V_VALS, kernels=KERNELS, print_iters=False, optimize_labels=None, cont=False, test_run=False,
				results_path=DEFAULT_RESULTS_PATH, figures_path=DEFAULT_FIGURES_PATH, datasets_path=DEFAULT_DATASETS_PATH,
				etas_path=DEFAULT_ETAS_PATH, num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, drop20=False,
				test_csv_filename=None):
		self.results_path = results_path
		self.figures_path = figures_path
		self.datasets_path = datasets_path
		self.etas_path = etas_path
		self.file_prefix = file_prefix
		self.cont=cont
		self.val_type = val_type
		self.users_as_tasks = users_as_tasks
		self.cluster_users = user_clusters
		self.drop20=drop20
		if test_csv_filename is not None:
			self.test_csv_filename = self.datasets_path + test_csv_filename
		else:
			self.test_csv_filename = None
		self.save_prefix = self.getSavePrefix(file_prefix, replace=cont)

		self.test_tasks = helper.loadPickledTaskList(datasets_path, file_prefix, "Test", fix_y=True)
		self.train_tasks = helper.loadPickledTaskList(datasets_path, file_prefix, "Train", fix_y=True)
		if self.val_type != 'cross':
			self.val_tasks = helper.loadPickledTaskList(datasets_path, file_prefix, "Val", fix_y=True)

		# print dataset sizes
		print "Num train points:", sum([len(t['Y']) for t in self.train_tasks])
		if self.val_type != 'cross':
			print "Num val points:", sum([len(t['Y']) for t in self.val_tasks])
		print "Num test points:", sum([len(t['Y']) for t in self.test_tasks])

   		if self.val_type != 'cross':
   			self.initializeMTMKLModel(self.train_tasks)
		else:
			self.classifier = None	

		self.n_feats = helper.calculateNumFeatsInTaskList(self.test_tasks)
		self.n_tasks = len(self.test_tasks)

		if optimize_labels is None:
			self.optimize_labels = ['tomorrow_Group_Happiness_Evening_Label', 'tomorrow_Group_Health_Evening_Label', 'tomorrow_Group_Calmness_Evening_Label']
		else:
			self.optimize_labels = optimize_labels

		self.c_vals = c_vals
		self.v_vals = v_vals
		self.kernels = kernels
		self.beta_vals=beta_vals
		self.regularizers = regularizers

		self.tolerance = tolerance
		self.max_iter = max_iter
		self.print_iters = print_iters

		if test_run:
			print "This is only a testing run. Using cheap settings to make it faster"
			self.c_vals = [100]
			self.beta_vals = [.01]
			self.kernels = ['linear']
			self.v_vals = [1.0]
			self.regularizers = ['L1']
			self.max_iter = 1

		self.calcNumSettingsDesired()

		#storing the results
		self.time_sum = 0
		if cont:
			self.val_results_df = pd.DataFrame.from_csv(self.results_path + self.save_prefix + '.csv')
			print '\nPrevious validation results df loaded. It has', len(self.val_results_df), "rows"
			self.started_from = len(self.val_results_df)
		else:
			self.val_results_df = pd.DataFrame()
			self.started_from = 0

		self.num_cross_folds = num_cross_folds
		if self.val_type == 'cross':
			helper.generateCrossValPickleFiles(self.datasets_path, self.file_prefix, self.num_cross_folds)
			#helper.addKeepIndicesToCrossValPickleFiles(self.datasets_path, self.file_prefix, self.num_cross_folds, .80)

	def getSavePrefix(self, file_prefix, replace=False):
		name_modifier = ""
		if '/' in file_prefix:
			if "NoLocation" in file_prefix:
				name_modifier = "-noloc"
			slash_loc = file_prefix.find('/')
			path_modifier = file_prefix[0:slash_loc+1]
			file_prefix = file_prefix[slash_loc+1:]
			self.file_prefix = file_prefix
			self.datasets_path += path_modifier

		dash_loc = file_prefix.find('-')

		if self.users_as_tasks:
			task_str = '_users'
		else:
			task_str = '_wellbeing'

		prefix = "MTMKL" + task_str + file_prefix[dash_loc:-1] + name_modifier
		
		if not replace:
			while os.path.exists(self.results_path + prefix + '.csv'):
				prefix = prefix + '2'
		return prefix

	def calcNumSettingsDesired(self):
		self.num_settings = len(self.c_vals) * len(self.beta_vals) * len(self.kernels)  \
						* len(self.v_vals) * len(self.regularizers)

	# use something like the following to test only one set of parameters:
	# wrapper.setParams(tau10s=[.05], tau20s=[.05], sigma_multipliers=[.1,.01])
	def setParams(self, c_vals=None, beta_vals=None, kernels=None, v_vals=None, regularizers=None):
		'''does not override existing parameter settings if the parameter is not set'''
		self.c_vals = c_vals if c_vals is not None else self.c_vals
		self.beta_vals = beta_vals if beta_vals is not None else self.beta_vals
		self.kernels = kernels if kernels is not None else self.kernels
		self.v_vals = v_vals if v_vals is not None else self.v_vals
		self.regularizers = regularizers if regularizers is not None else self.regularizers

	def settingAlreadyDone(self, C, beta, kernel, v, regularizer):
		if kernel == 'linear':
			if len(self.val_results_df[(self.val_results_df['C']== C) & \
										(self.val_results_df['kernel']== kernel) & \
										(self.val_results_df['v']== v) & \
										(self.val_results_df['regularizer']== regularizer)]) > 0:
				print "setting already tested"
				return True
			else:
				return False
		else:
			if len(self.val_results_df[(self.val_results_df['C']== C) & \
										(self.val_results_df['beta']== beta) & \
										(self.val_results_df['kernel']== kernel) & \
										(self.val_results_df['v']== v) & \
										(self.val_results_df['regularizer']== regularizer)]) > 0:
				print "setting already tested"
				return True
			else:
				return False

	def initializeMTMKLModel(self, train_tasks, verbose=False):
		if USE_TENSORFLOW:
			self.classifier = mtmkl_tf.MTMKL(train_tasks,verbose=verbose,tol=self.tolerance, debug=False, max_iter=self.max_iter)
		else:
			self.classifier = mtmkl.MTMKL(train_tasks,verbose=verbose,tol=self.tolerance, debug=False, max_iter=self.max_iter, drop20PercentTrainingData=self.drop20)

	def setClassifierToSetting(self, C, beta, kernel, v, regularizer):
		self.classifier.setAllSettings(C, v, kernel, beta, regularizer)

	#must have called setValData for now
	def initializeAndTrainMTMKL(self, train_tasks, C, beta, kernel, v, regularizer, verbose=False):
		self.initializeMTMKLModel(train_tasks,verbose=verbose)
		self.setClassifierToSetting(C, beta, kernel, v, regularizer)
		converged = self.classifier.train()
		return converged

	def getValidationResults(self, results_dict, C, beta, kernel, v, regularizer):
		converged = self.initializeAndTrainMTMKL(self.train_tasks, C, beta, kernel, v, regularizer)

		if self.users_as_tasks:
			if not converged:
				val_acc = np.nan
				val_auc = np.nan
			else:
				val_acc, val_auc = self.classifier.getAccuracyAucAllTasks(self.val_tasks)
			results_dict['val_acc'] = val_acc
			results_dict['val_auc'] = val_auc
		else:
			accs = []
			aucs = []
			for t in range(self.n_tasks):
				if not converged:
					acc = np.nan
					auc = np.nan
				else:
					acc, auc = self.classifier.getAccuracyAucOnOneTask(self.val_tasks, t)
				task_name = self.val_tasks[t]['Name']
				results_dict['TaskAcc-' + helper.getFriendlyLabelName(task_name)] = acc
				results_dict['TaskAuc-' + helper.getFriendlyLabelName(task_name)] = auc
				if self.cluster_users or task_name in self.optimize_labels:
					accs.append(acc)
					aucs.append(auc)
			results_dict['val_acc'] = np.mean(accs)
			results_dict['val_auc'] = np.mean(aucs)
		return results_dict

	def getCrossValidationResults(self, results_dict, C, beta, kernel, v, regularizer, save_plots=False,print_per_fold=True):
		all_acc = []
		all_auc = []
		all_f1 = []
		all_precision = []
		all_recall = []
		if not self.users_as_tasks:	
			per_task_accs = [[] for i in range(self.n_tasks)]
			per_task_aucs = [[] for i in range(self.n_tasks)]
			per_task_f1 = [[] for i in range(self.n_tasks)]
			per_task_precision = [[] for i in range(self.n_tasks)]
			per_task_recall = [[] for i in range(self.n_tasks)]

		for f in range(self.num_cross_folds):
			train_tasks, val_tasks = helper.loadCrossValData(self.datasets_path, self.file_prefix, f, reshape=False, fix_y=True)
			converged = self.initializeAndTrainMTMKL(train_tasks, C, beta, kernel, v, regularizer)
			if not converged:
				all_acc.append(np.nan)
				all_auc.append(np.nan)
				all_f1.append(np.nan)
				all_precision.append(np.nan)
				all_recall.append(np.nan)
				continue
	
			# Get results!
			fold_preds = []
			fold_true_y = []
			for t in range(self.n_tasks):
				preds = self.classifier.predictOneTask(val_tasks,t)
				true_y = list(val_tasks[t]['Y'].flatten())

				if not self.users_as_tasks:
					# save the per-task results
					t_acc, t_auc, t_f1, t_precision, t_recall = helper.computeAllMetricsForPreds(preds, true_y)
					per_task_accs[t].append(t_acc)
					per_task_aucs[t].append(t_auc)
					per_task_f1[t].append(t_f1)
					per_task_precision[t].append(t_precision)
					per_task_recall[t].append(t_recall)
					if print_per_fold: print "Fold", f, "Task", val_tasks[t]['Name'], "acc", t_acc, "auc", t_auc, "f1", t_f1, "precision",t_precision,"recall",t_recall

				fold_preds.extend(preds)
				fold_true_y.extend(true_y)


			acc, auc, f1, precision, recall = helper.computeAllMetricsForPreds(fold_preds, fold_true_y)
			all_acc.append(acc)
			all_auc.append(auc)
			all_f1.append(f1)
			all_precision.append(precision)
			all_recall.append(recall)
			if print_per_fold: print "Fold", f, "acc", acc, "auc", auc, "f1", f1, "precision",precision,"recall",recall

		print "accs for all folds", all_acc
		print "aucs for all folds", all_auc
		
		# Add results to the dictionary
		results_dict['val_acc'] = np.nanmean(all_acc)
		results_dict['val_auc'] = np.nanmean(all_auc)
		results_dict['val_f1'] = np.nanmean(all_f1)
		results_dict['val_precision'] = np.nanmean(all_precision)
		results_dict['val_recall'] = np.nanmean(all_recall)

		# Add per-task results to the dictionary
		if not self.users_as_tasks:
			for t in range(self.n_tasks):
				task_name = val_tasks[t]['Name']
				results_dict['TaskAcc-' + helper.getFriendlyLabelName(task_name)] = np.nanmean(per_task_accs[t])
				results_dict['TaskAuc-' + helper.getFriendlyLabelName(task_name)] = np.nanmean(per_task_aucs[t])
				results_dict['TaskF1-' + helper.getFriendlyLabelName(task_name)] = np.nanmean(per_task_f1[t])
				results_dict['TaskPrecision-' + helper.getFriendlyLabelName(task_name)] = np.nanmean(per_task_precision[t])
				results_dict['TaskRecall-' + helper.getFriendlyLabelName(task_name)] = np.nanmean(per_task_recall[t])

		return results_dict

	def testOneSetting(self, C, beta, kernel, v, regularizer):
		if self.cont:
			if self.settingAlreadyDone(C, beta, kernel, v, regularizer):
				return

		t0 = time()
		
		results_dict = {'C':C, 'beta': beta, 'kernel':kernel, 'v':v, 'regularizer':regularizer}
		print results_dict
		
		if self.val_type == 'cross':
			results_dict = self.getCrossValidationResults(results_dict, C, beta, kernel, v, regularizer)
		else:
			results_dict = self.getValidationResults(results_dict, C, beta, kernel, v, regularizer)
		
		self.val_results_df = self.val_results_df.append(results_dict,ignore_index=True)
		
		print "\n", self.val_results_df.tail(n=1)
		t1 = time()
		this_time = t1 - t0
		print "It took", this_time, "seconds to obtain this result"

		self.time_sum = self.time_sum + this_time

		self.printTimeEstimate()
		sys.stdout.flush()

		#output the file every few iterations for safekeeping 
		if len(self.val_results_df) % SAVE_RESULTS_EVERY_X_TESTS == 0:
			self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

	def printTimeEstimate(self):
		num_done = len(self.val_results_df)-self.started_from
		num_remaining = self.num_settings - num_done - self.started_from
		avg_time = self.time_sum / num_done
		total_secs_remaining = int(avg_time * num_remaining)
		hours = total_secs_remaining / 60 / 60
		mins = (total_secs_remaining % 3600) / 60
		secs = (total_secs_remaining % 3600) % 60

		print "\n", num_done, "settings processed so far,", num_remaining, "left to go"
		print "Estimated time remaining:", hours, "hours", mins, "mins", secs, "secs"

	def sweepAllParameters(self):
		print "\nSweeping all parameters!"
		
		self.calcNumSettingsDesired()
		print "\nYou have chosen to test a total of", self.num_settings, "settings"
		sys.stdout.flush()

		#sweep all possible combinations of parameters
		for C in self.c_vals:
			for v in self.v_vals:
				for regularizer in self.regularizers:
					for kernel in self.kernels:
						if kernel == 'linear':
							self.testOneSetting(C, np.nan, kernel, v, regularizer)
						else:
							for beta in self.beta_vals:
								self.testOneSetting(C, beta, kernel, v, regularizer)
		self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

	def run(self):
		self.sweepAllParameters()
		return self.findBestSetting(criteria='AUC')


	def findBestSetting(self, criteria="accuracy", minimize=False, save_final_results=True):
		if criteria=="accuracy":
			search_col = 'val_auc'
		elif criteria=="AUC":
			search_col = 'val_auc'

		results = self.val_results_df[search_col].tolist()
		if minimize:
			best_result = min(results)
			opt_word = "minimized"
		else:
			best_result = max(results)
			opt_word = "maximized"
		best_idx = results.index(best_result)

		print "BEST SETTING!"
		print "Settings which", opt_word, "the", criteria, "were:"
		print self.val_results_df.iloc[best_idx]

		if save_final_results:
			self.getFinalResultsAndSave(self.val_results_df.iloc[best_idx])
		else:
			return self.val_results_df.iloc[best_idx]

	def getFinalResultsAndSave(self, results_dict):
		print "\nRetraining on full training data with the best settings..."
		self.drop20=False
		self.initializeAndTrainMTMKL(self.train_tasks, results_dict['C'], results_dict['beta'], 
									results_dict['kernel'], results_dict['v'], results_dict['regularizer'], 
									verbose=True)
		
		print "\nEvaluating results on held-out test set!! ..."
		all_preds = []
		all_true_y = []
		per_task_accs = [np.nan] * self.n_tasks
		per_task_aucs = [np.nan] * self.n_tasks
		per_task_f1 = [np.nan] * self.n_tasks
		per_task_precision = [np.nan] * self.n_tasks
		per_task_recall = [np.nan] * self.n_tasks
		for t in range(self.n_tasks):
			preds = self.classifier.predictOneTask(self.test_tasks,t)
			true_y = list(self.test_tasks[t]['Y'].flatten())

			if len(preds)==0 or len(true_y) == 0:
				print "no y for task", t, "... skipping"
				continue
				
			all_preds.extend(preds)
			all_true_y.extend(true_y)

			# save the per-task results
			t_acc, t_auc, t_f1, t_precision, t_recall = helper.computeAllMetricsForPreds(preds, true_y)
			per_task_accs[t] = t_acc
			per_task_aucs[t] = t_auc
			per_task_f1[t] = t_f1
			per_task_precision[t] = t_precision
			per_task_recall[t] = t_recall

		print "\nPlotting cool stuff about the final model..."
		self.saveImagePlot(self.classifier.eta, 'Etas')
		pd.DataFrame(self.classifier.eta).to_csv(self.etas_path + self.save_prefix + "-etas.csv")

		print "\tHELD OUT TEST METRICS COMPUTED BY APPENDING ALL PREDS"
		acc, auc, f1, precision, recall = helper.computeAllMetricsForPreds(all_preds, all_true_y)
		print '\t\tAcc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall

		print "\n\tHELD OUT TEST METRICS COMPUTED BY AVERAGING OVER TASKS"
		avg_acc = np.nanmean(per_task_accs)
		avg_auc = np.nanmean(per_task_aucs)
		avg_f1 = np.nanmean(per_task_f1)
		avg_precision = np.nanmean(per_task_precision)
		avg_recall = np.nanmean(per_task_recall)
		print '\t\tAcc:', avg_acc, 'AUC:', avg_auc, 'F1:', avg_f1, 'Precision:', avg_precision, 'Recall:', avg_recall

		print "\n\tHELD OUT TEST METRICS COMPUTED FOR EACH TASK"
		if not self.users_as_tasks:
			for t in range(self.n_tasks):
				task_name = self.test_tasks[t]['Name']
				task_name=helper.getFriendlyLabelName(task_name)
				print "\t\t", task_name, "- Acc:", per_task_accs[t], "AUC:", per_task_aucs[t], 'F1:', per_task_f1[t], 'Precision:', per_task_precision[t], 'Recall:', per_task_recall[t]

		if self.test_csv_filename is not None:
			print "\tSAVING HELD OUT PREDICITONS"
			if 'Big5GenderKMeansCluster' in self.file_prefix:
				task_column = 'Big5GenderKMeansCluster'
				tasks_are_ints = True
				label_name = helper.getFriendlyLabelName(self.file_prefix)
				wanted_label = helper.getOfficialLabelName(label_name)
				predictions_df = helper.get_test_predictions_for_df_with_task_column(
						self.classifier.predict_01, self.test_csv_filename, task_column, self.test_tasks, 
						wanted_label=wanted_label, num_feats_expected=np.shape(self.test_tasks[0]['X'])[1], 
						label_name=label_name, tasks_are_ints=tasks_are_ints)
			elif not self.users_as_tasks:
				predictions_df = helper.get_test_predictions_for_df_with_no_task_column(self.classifier.predict_01, 
					self.test_csv_filename, self.test_tasks, num_feats_expected=np.shape(self.test_tasks[0]['X'])[1])
			else:
				print "Error! Cannot determine what type of model you are training and therefore cannot save predictions."
				return
			predictions_df.to_csv(self.results_path + "Preds-" + self.save_prefix + '.csv')
		else:
			print "Uh oh, the test csv filename was not set, can't save test preds"

	def saveImagePlot(self, matrix, name):
		plt.figure()
		plt.imshow(matrix)
		plt.savefig(self.figures_path + self.save_prefix + "-" + name + ".eps")
		plt.close()

	

if __name__ == "__main__":
	print "MTMKL MODEL SELECTION"
	print "\tThis code will sweep a set of parameters to find the ideal settings for MTMLK for a single dataset"

	if len(sys.argv) < 3:
		print "Error: usage is python MTMKLWrapper.py <data file> <test type> <continue>"
		print "\t<file prefix>: e.g. datasetTaskList-Discard-Future-Group_ - program will look in the following directory for this file", DEFAULT_DATASETS_PATH
		print "\t<test type>: type 'users' for users as tasks, 'wellbeing' for wellbeing measures as tasks, or 'clusters' for user clusters as tasks"
		print "\t<continue>: optional. If 'True', the wrapper will pick up from where it left off by loading a previous validation results file"
		print "\t<csv file for testing>: optional. If you want to get the final test results, provide the name of a csv file to test on"
		sys.exit()
	filename= sys.argv[1] #get data file from command line argument
	print "\nLoading dataset", DEFAULT_DATASETS_PATH + filename
	print ""

	if sys.argv[2] == 'users':
		users_as_tasks = True
		cluster_users = False
		print "Okay, treating users as tasks. Will not print per-task results"
	elif sys.argv[2] == 'wellbeing':
		users_as_tasks = False
		cluster_users = False
		print "Okay, treating wellbeing measures as tasks. Will save and print per-task results"
	elif sys.argv[2] == 'clusters':
		users_as_tasks = False
		cluster_users = True
		print "Okay, treating user clusters as tasks. Will save and print per-task results and optimize for accuracy over all clusters."
		
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

	if USE_TENSORFLOW:
		print "\nWill use the TENSORFLOW version of the code\n"

	wrapper = MTMKLWrapper(filename, users_as_tasks=users_as_tasks, user_clusters=cluster_users, cont=cont,
						   test_csv_filename=csv_test_file)
	
	print "\nThe following parameter settings will be tested:"
	print "\tCs:            \t", wrapper.c_vals
	print "\tbetas:         \t", wrapper.beta_vals
	print "\tkernels:       \t", wrapper.kernels
	print "\tvs:            \t", wrapper.v_vals
	print "\tregularizers:  \t", wrapper.regularizers

	print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'
	print "\nThe validation and testing figures will be saved in:", wrapper.figures_path + wrapper.save_prefix

	wrapper.run()
