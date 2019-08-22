"""Performs a hyperparameter sweep for the Single Task Learning (STL) neural
network."""
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os
import pickle
import copy
from time import time

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_RESULTS_PATH = '/Your/path/here/'
DEFAULT_DATASETS_PATH = '/Your/path/here/'
DEFAULT_FIGURES_PATH = '/Your/path/here/'

import tensorFlowNetwork as tfnet
import helperFuncs as helper

DEFAULT_VAL_TYPE = 'cross'
DEFAULT_NUM_CROSS_FOLDS = 5
SAVE_RESULTS_EVERY_X_TESTS = 1

def reloadFiles():
	reload(helper)
	reload(tfnet)
	tfnet.reloadHelper()

class TensorFlowSTLWrapper:
	
	def __init__(self, dataset_name, target_label, users_as_tasks=True, test_steps=9001, val_output_file=None, 
				val_type=DEFAULT_VAL_TYPE, cont=False, results_path=DEFAULT_RESULTS_PATH, 
				datasets_path=DEFAULT_DATASETS_PATH, figures_path=DEFAULT_FIGURES_PATH, architectures=None, 
				num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, test_run=False, redo_test=False):
		self.datasets_path = datasets_path
		self.cont = cont
		self.val_type = val_type
		self.num_cross_folds = num_cross_folds
		self.test_steps = test_steps
		self.redo_test = redo_test
		self.users_as_tasks = users_as_tasks
		self.target_label = target_label
		if self.users_as_tasks:
			self.results_path = results_path + 'STL-OneModelPerUser/'
			self.figures_path = figures_path + 'STL-OneModelPerUser/'
		else:
			self.results_path = results_path + 'STL-Wellbeing/'
			self.figures_path = figures_path + 'STL-Wellbeing/'
		self.save_prefix = self.getSavePrefix(dataset_name, target_label, replace=cont)

		self.dataset_name = dataset_name 
		self.data_df = pd.DataFrame.from_csv(self.datasets_path + self.dataset_name)
		self.wanted_feats = [x for x in self.data_df.columns.values if x != 'user_id' and x != 'timestamp' and x!= 'dataset' and '_Label' not in x]
		if self.users_as_tasks:
			self.wanted_labels = [target_label]
			self.n_tasks = len(self.data_df['user_id'].unique())
		else:
			self.wanted_labels = [x for x in self.data_df.columns.values if '_Label' in x and 'tomorrow_' in x and 'Evening' in x and 'Alertness' not in x and 'Energy' not in x]
			self.n_tasks = len(self.wanted_labels)

		#parameters that can be tuned:
		self.l2_regularizers = [1e-2, 1e-4]
		self.dropout = [True, False]
		self.decay = [True]
		self.decay_steps = [10000]
		self.decay_rates = [0.95]
		self.optimizers = [tf.train.AdamOptimizer] #[tf.train.AdagradOptimizer,  tf.train.GradientDescentOptimizer
		self.train_steps =[4001]
		self.batch_sizes = [5,10,20]
		self.learning_rates = [.01, .001]
		self.architectures = [[100],[50,5],[100,10]] if architectures is None else architectures

		self.test_run = test_run
		if test_run:
			print "This is only a testing run. Using cheap settings to make it faster"
			self.l2_regularizers = [1e-2]
			self.dropout = [True]
			self.decay = [True]
			self.decay_steps = [10000]
			self.decay_rates = [0.95]
			self.optimizers = [tf.train.AdamOptimizer] #[tf.train.AdagradOptimizer,  tf.train.GradientDescentOptimizer
			self.train_steps =[1001]
			self.batch_sizes = [10]
			self.learning_rates = [.001]
			self.architectures = [[100],[50,5]] if architectures is None else architectures

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

		# store for computing the accuracy/auc the unfair way
		self.cumulative_test_preds = []
		self.cumulative_test_true = []

	def getSavePrefix(self, file_name, target_label, replace=False):
		if '/' in file_name:
			slash_loc = file_name.find('/')
			file_name = file_name[slash_loc:]
		dash_loc = file_name.find('-')
		if self.users_as_tasks:
			task_name = "tfSTLUsers"
			label_name = '-' + helper.getFriendlyLabelName(target_label)
		else:
			task_name = "tfSTLWellbeing"
			label_name = ""
		prefix = task_name + file_name[dash_loc:-4] + label_name 
		if not replace:
			while os.path.exists(self.results_path + prefix + '.csv'):
				prefix = prefix + '2'
		return prefix

	def calcNumSettingsDesired(self):
		self.num_settings = len(self.l2_regularizers) * len(self.learning_rates) * len(self.dropout) * len(self.decay) \
						* len(self.batch_sizes)  * len(self.optimizers) * len(self.train_steps) * len(self.architectures)
		if True in self.decay and (len(self.decay_steps) > 1 or len(self.decay_rates) > 1):
			self.num_settings = num_settings * ((len(self.decay_steps) * len(self.decay_rates)) / 2.0) 

	# use something like the following to test only one set of parameters:
	# wrapper.setParams(l2_regularizers=[1e-4, learning_rates=[.01], dropout=[True], decay=[True], batch_sizes=[50], optimizers=[tf.train.GradientDescentOptimizer])
	def setParams(self, l2_regularizers=None, learning_rates=None, dropout=None, 
				decay=None, decay_steps=None, decay_rates=None, batch_sizes=None,
				optimizers=None, train_steps=None):
		'''does not override existing parameter settings if the parameter is not set'''
		self.l2_regularizers = l2_regularizers if l2_regularizers is not None else self.l2_regularizers
		self.learning_rates = learning_rates if learning_rates is not None else self.learning_rates
		self.dropout= dropout if dropout is not None else self.dropout
		self.decay= decay if decay is not None else self.decay
		self.decay_steps= decay_steps if decay_steps is not None else self.decay_steps
		self.decay_rates= decay_rates if decay_rates is not None else self.decay_rates
		self.batch_sizes = batch_sizes if batch_sizes is not None else self.batch_sizes
		self.optimizers = optimizers if optimizers is not None else self.optimizers

	def settingAlreadyDone(self, task):
		if len(self.val_results_df[(self.val_results_df['task_name']== task)]) > 0:
			print "setting already tested"
			return True
		else:
			return False

	def getResultsDictFromRow(self,row_df):
		best_results_dict = dict()
		for col in row_df.columns.values:
			best_results_dict[col] = row_df[col].tolist()[0]

		for arch in self.architectures:
			if str(arch) == best_results_dict['hidden_layers']:
				best_results_dict['hidden_layers'] = arch

		for opt_func in self.optimizers:
			if str(opt_func) == best_results_dict['optimizer']:
				best_results_dict['optimizer'] = opt_func

		return best_results_dict

	def constructNetwork(self, hidden_layers):
		connections = ['full'] * (len(hidden_layers)+1)
		self.net.setUpNetworkStructure(hidden_layers,connections)

	def sweepParametersForOneTask(self, task_name, target_label):
		if self.users_as_tasks:
			task_df = self.data_df[self.data_df['user_id'] == task_name]
		else:
			task_df = self.data_df
		self.net = tfnet.TensorFlowNetwork(task_df, copy.deepcopy(self.wanted_feats), self.wanted_labels, verbose=False, val_type=self.val_type)

		if len(self.net.train_X) == 0 or len(self.net.train_y) == 0:
			print "No training data for this task!"
			return dict()
		if len(self.net.test_X) == 0:
			print "No testing data for this task! Skipping",
			return dict()
		if np.shape(self.net.train_X)[1] == 0: 
			print "All columns were null, this task has no features left!"
			return dict()
		if len(self.net.train_X) != len(self.net.train_y):
			print "Unequal length of X and Y dataframe!"
			return dict()

		df = pd.DataFrame()
		
		#sweep all possible combinations of parameters
		print "...sweeping all parameters for this task..."
		for hidden_layers in self.architectures:
			for l2_beta in self.l2_regularizers:
				for lrate in self.learning_rates:
					for dropout in self.dropout:
						for bsize in self.batch_sizes:
							for opt in self.optimizers:
								for tsteps in self.train_steps:
									for decay in self.decay:
										if decay:
											for dsteps in self.decay_steps:
												for drate in self.decay_rates:
													results_dict = self.testOneSettingForOneTask(hidden_layers, l2_beta, lrate, dropout, decay, dsteps, drate, bsize, opt, tsteps)
													df = df.append(results_dict ,ignore_index=True)
										else:
											#decay steps and decay rate don't matter if decay is set to false
											results_dict = self.testOneSettingForOneTask(hidden_layers, l2_beta, lrate, dropout, decay, 10000, 0.95, bsize, opt, tsteps)
											df = df.append(results_dict ,ignore_index=True)
		
		accuracies = df['val_acc'].tolist()
		max_acc = max(accuracies)
		max_idx = accuracies.index(max_acc)

		best_results_dict = df.iloc[max_idx]

		#retrain with the best settings

		test_acc, test_auc, test_preds = self.getFinalResultsForTask(best_results_dict)
		self.cumulative_test_preds.extend(test_preds)
		self.cumulative_test_true.extend(self.net.test_X)

		best_results_dict['test_acc'] = test_acc
		best_results_dict['test_auc'] = test_auc
		return best_results_dict

	def find_best_setting(self, task):
		df = self.val_results_df[self.val_results_df['task_name']==task]
		accuracies = df['val_acc'].tolist()
		max_acc = max(accuracies)
		max_idx = accuracies.index(max_acc)

		best_results_dict = df.iloc[max_idx]
		return helper.fixSettingDictLoadedFromResultsDf(best_results_dict)

	def getFinalResultsForTask(self, setting_dict):
		if self.users_as_tasks:
			task_df = self.data_df[self.data_df['user_id'] == setting_dict['task_name']]
			target_label = [self.target_label]
		else:
			task_df = self.data_df
			target_label = [helper.getOfficialLabelName(setting_dict['task_name'])]
		self.net = tfnet.TensorFlowNetwork(task_df, copy.deepcopy(self.wanted_feats),target_label, verbose=False, val_type=self.val_type)
		self.net.setParams(l2_beta=setting_dict['l2_beta'], initial_learning_rate=setting_dict['learning_rate'], decay=setting_dict['decay'], 
							decay_steps=setting_dict['decay_steps'], decay_rate=setting_dict['decay_rate'], batch_size=setting_dict['batch_size'],
							optimizer=setting_dict['optimizer'], dropout=setting_dict['dropout'])
		self.constructNetwork(setting_dict['hidden_layers'])

		self.net.setUpGraph()
		preds = self.net.runGraph(self.test_steps, print_test=True, return_test_preds=True)

		preds_df = self.net.get_preds_for_df()
		label_name = setting_dict['task_name']
		preds_df.to_csv(self.results_path + "Preds-" + self.save_prefix + label_name + '.csv')
		print "Preds df saved to", self.results_path + "Preds-" + self.save_prefix + label_name + '.csv'

		return self.net.final_test_results['acc'], self.net.final_test_results['auc'], preds

	def testOneSettingForOneTask(self, hidden_layers, l2_beta, lrate, dropout, decay, dsteps, drate, bsize, opt, tsteps):
		self.net.setParams(l2_beta=l2_beta, initial_learning_rate=lrate, decay=decay, 
							decay_steps=dsteps, decay_rate=drate, batch_size=bsize,
							optimizer=opt, n_steps=tsteps, dropout=dropout)
		self.constructNetwork(hidden_layers)
		if self.val_type == 'cross':
			val_acc, val_auc, val_f1, val_prec, val_recall = self.net.trainAndCrossValidate()
		else:
			val_acc, val_auc, val_f1, val_prec, val_recall = self.net.trainAndValidate()

		results_dict = {'hidden_layers':hidden_layers, 'l2_beta': l2_beta, 'learning_rate': lrate, 
						'dropout': dropout, 'decay': decay, 'decay_steps': dsteps, 
						'decay_rate': drate, 'batch_size': bsize, 
						'optimizer': opt, 'val_acc': val_acc, 'val_auc':val_auc}

		return results_dict

	
	def runOneTask(self, task, target_label):
		print "\nRunning task", task
		if self.cont:
			if self.settingAlreadyDone(task):
				if self.redo_test:
					self.redoTestResult(task)
				best_setting = self.find_best_setting(task)
				print "The setting that produced the best validation results for task", task, "was:"
				print best_setting
				self.getFinalResultsForTask(best_setting)
				return 

		t0 = time()
		
		results_dict = self.sweepParametersForOneTask(task, target_label)
		results_dict['task_name'] = task
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
		num_remaining = self.n_tasks - num_done - self.started_from
		avg_time = self.time_sum / num_done
		total_secs_remaining = int(avg_time * num_remaining)
		hours = total_secs_remaining / 60 / 60
		mins = (total_secs_remaining % 3600) / 60
		secs = (total_secs_remaining % 3600) % 60

		print "\n", num_done, "settings processed so far,", num_remaining, "left to go"
		print "Estimated time remaining:", hours, "hours", mins, "mins", secs, "secs"

	def run(self):
		print "\nYou have chosen to test a total of", self.num_settings, "settings for each task"
		print "There are", self.n_tasks, "tasks, meaning you are training a total of..."
		print "\t", self.num_settings * self.n_tasks, "neural networks!!"
		sys.stdout.flush()

		if self.users_as_tasks:
			tasks = self.data_df['user_id'].unique()
		else:
			tasks = [helper.getFriendlyLabelName(x) for x in self.wanted_labels]

		i = 0
		for t in range(len(tasks)):
			if self.users_as_tasks:
				self.runOneTask(tasks[i], self.target_label)
			else:
				self.runOneTask(tasks[i], self.wanted_labels[i])
			if self.test_run and i > 2:
				break
			i += 1
		self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

		if self.users_as_tasks:
			print "\n\nFINAL RESULTS - Averaging individual models:"
			print "\tValidation set: Accuracy =", np.nanmean(self.val_results_df['val_acc']), "AUC = ", np.nanmean(self.val_results_df['val_auc'])
			print "\tTest set: Accuracy =", np.nanmean(self.val_results_df['test_acc']), "AUC = ", np.nanmean(self.val_results_df['test_auc'])
			print ""
			print "FINAL RESULTS - Aggregating predictions of individual models"
			agg_auc = helper.computeAuc(self.cumulative_test_preds, self.cumulative_test_true)
			agg_acc = helper.getBinaryAccuracy(self.cumulative_test_preds, self.cumulative_test_true)
			print "\tTest set: Accuracy =", agg_acc, "AUC = ", agg_auc


if __name__ == "__main__":
	print "TENSOR FLOW STL MODEL SELECTION"
	print "\tFor each tasl individually, this code will sweep a set of network architectures and parameters to find the ideal settings"
	print "\tIt will record the settings, validation and test results for each user"

	if len(sys.argv) < 3:
		print "Error: usage is python tensorFlowWrapperSTL.py <data file> <task type> <target label> <continue> <redo test>"
		print "\t<data file>: e.g. dataset-Simple-Group.csv - program will look in the following directory for this file", DEFAULT_DATASETS_PATH
		print "\t<task type>: type 'users' for users as tasks, or 'wellbeing' for wellbeing measures as tasks"
		print "\t<target label>: Only required for users-as-tasks. Enter the name of the label you would like classify on. E.g. tomorrow_Group_Happiness_Evening_Label."
		print "\t<continue>: optional. If 'True', the neural net will pick up from where it left off by loading a previous validation results file"
		print "\t<redo test>: optional. If 'redo' the neural net will go through the saved validation results file and compute test predictions for each user for each setting. It will collect all the preds and only compute AUC at the end"
		sys.exit()
	filename= sys.argv[1] #get data file from command line argument
	task_type = sys.argv[2]
	if len(sys.argv) >= 4:
		target_label = sys.argv[3]
		print "Classifying on target label:", target_label
	else:
		target_label = None

	print "\nLoading dataset", DEFAULT_DATASETS_PATH + filename
	if task_type == 'wellbeing':
		users_as_tasks = False
		print "Performing wellbeing-as-tasks classification\n"
	else:
		users_as_tasks = True
		print "Performing users-as-tasks classification\n"

	if len(sys.argv) >= 5 and sys.argv[4] == 'True':
		cont = True
		print "Okay, will continue from a previously saved validation results file for this problem"
	else:
		cont = False
	print ""

	redo = False
	if len(sys.argv) >= 6 and sys.argv[5] == 'redo':
		redo = True
		print "Okay, will redo all the test results to get a better AUC"


	wrapper = TensorFlowSTLWrapper(filename, target_label=target_label, users_as_tasks=users_as_tasks, cont=cont,
										results_path=DEFAULT_RESULTS_PATH, datasets_path=DEFAULT_DATASETS_PATH, figures_path=DEFAULT_FIGURES_PATH)
	
	if not redo:
		print "\nThe following parameter settings will be tested for each task:"
		print "\tl2_regularizers:  \t", wrapper.l2_regularizers
		print "\tlearning_rates:   \t", wrapper.learning_rates
		print "\tdropout:          \t", wrapper.dropout
		print "\tdecay:            \t", wrapper.decay
		print "\tdecay_steps:      \t", wrapper.decay_steps
		print "\tdecay_rates:      \t", wrapper.decay_rates
		print "\tbatch_sizes:      \t", wrapper.batch_sizes
		print "\toptimizers:       \t", wrapper.optimizers
		print "\ttrain_steps:      \t", wrapper.train_steps

		print "\nThe following network structures will be tested:"
		print "\t", wrapper.architectures

		print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'
		print "\nThe validation accuracy figures will be saved in:", wrapper.figures_path + wrapper.save_prefix + '.eps'

		wrapper.run()
	else:
		wrapper.redoAllTestsResults()

