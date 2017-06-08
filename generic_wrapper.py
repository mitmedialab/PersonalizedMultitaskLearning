import numpy as np
import pandas as pd
import os
import sys
import copy
from time import time

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

DEFAULT_VALIDATION_TYPE = 'cross' #'val'
DEFAULT_NUM_CROSS_FOLDS = 5

import helperFuncs as helper

def reload_dependencies():
	reload(helper)

# This optimizes parameters individually for each task

class STLWrapper:
	""" WARNING: This code only deals with input files in the form of pickled task lists,
	and only implements cross validation."""
	def __init__(self, file_prefix, users_as_tasks=False, cont=False, classifier_name='LSSVM', 
				num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, main_directory=DEFAULT_MAIN_DIRECTORY, 
				datasets_path='Data/Datasets/Discard20/', cant_train_with_one_class=True,
				check_test=False, save_results_every_nth=3, test_csv_filename=None):
		""" Initializes the parent model with fields useful for all child wrapper classes

		Args:
			file_prefix: The first portion of the name of a set of pickled task lists, e.g.
				'datasetTaskList-Discard-Future-Group_'
			users_as_tasks: A boolean. If true, will assume there are many tasks and each task
				is one person. Will not print results per task. 
			cont: A boolean. If true, will try to load a saved results .csv and continue 
				training on the next unfinished result.
			classifier_name: String name of the classifier trained. Used to know where to save
				results.
			num_cross_folds: An integer number of folds to use in cross validation.
			main_directory: The path to the main dropbox directory which contains the results and
				data directories.
			datasets_path: The path from the main dropbox to the datasets directory.
			cant_train_with_one_class: A boolean. If true, if the model encounters a task with 
				only one type of label in the training data, it will just predict the most 
				frequent class. 
			check_test: A boolean. If true, will evaluate final results on held-out test set 
				after running.
			save_results_every_nth: An integer representing the number of settings to test before
				writing the results df to a csv file.
		"""
		# memorize arguments and construct paths
		self.main_directory = main_directory
		self.classifier_name = classifier_name
		self.results_path = main_directory + 'Results/' + classifier_name + '/'
		self.figures_path = main_directory + 'Figures/' + classifier_name + '/'
		self.datasets_path = main_directory + datasets_path
		self.cont = cont
		self.users_as_tasks = users_as_tasks
		self.cant_train_with_one_class = cant_train_with_one_class
		self.check_test = check_test
		self.save_results_every_nth = save_results_every_nth
		self.file_prefix = file_prefix
		self.save_prefix = self.get_save_prefix(file_prefix, replace=cont)
		if test_csv_filename is not None:
			self.test_csv_filename = self.datasets_path + test_csv_filename
		else:
			self.test_csv_filename = None

		self.params = {}
		self.define_params()

		self.load_data()

		self.calc_num_param_settings()
		self.construct_list_of_params_to_test()

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
		helper.generateCrossValPickleFiles(self.datasets_path, self.file_prefix, self.num_cross_folds)

	# These functions need to be overwritten by the child class
	def define_params(self):
		""" This function should set self.params to a dict where they keys represent names of parameters
			to test (e.g. for SVM, 'C') as they should be saved to the val_results_df, and the values of 
			self.params should be a list of values for the parameter that need to be tested. An example 
			dict:
				self.params['C'] = [1,10,100]
				self.params['beta'] = [.001, .01, .1]
		"""
		print "Error! define_params should be overwritten in child class"
		raise NotImplementedError

	def train_and_predict_task(self, t, train_X, train_y, eval_X, param_dict):
		print "Error! train_model_for_task should be overwritten in child class"
		raise NotImplementedError

	def predict_task(self, X, t):
		print "Error! predict_task should be overwritten in child class"
		raise NotImplementedError

	def calc_num_param_settings(self):
		self.num_settings = self.n_tasks
		for key in self.params:
			self.num_settings = self.num_settings * len(self.params[key])

	def construct_list_of_params_to_test(self):
		"""Will make a class level variable that is a list of parameter dicts.
		Each entry in the list is a dict of parameter settings, 
		eg. {'C'=1.0, 'beta'=.01, ...}. All tasks can use this list to train
		against all settings."""
		self.list_of_param_settings = []
		self.recurse_and_append_params(copy.deepcopy(self.params), {})

	def recurse_and_append_params(self, param_settings_left, this_param_dict, debug=False):
		"""param_settings_left is a dictionary of lists. The keys are parameters
		(like 'C'), the values are the list of settings for those parameters that 
		need to be tested (like [1.0, 10.0, 100.0]). this_param_dict is a dictionary 
		containing a single setting for each parameter. If a parameter is not in 
		this_param_dict's keys, a setting for it has not been chosen yet.
		
		Performs breadth-first-search"""
		if debug: print "Working on a parameter dict containing", this_param_dict
		for key in self.params.keys():
			if key in this_param_dict:
				continue
			else:
				this_setting = param_settings_left[key].pop()
				if debug: print "Popped", key, "=", this_setting, "off the params left"
				if len(param_settings_left[key]) > 0:
					if debug: print "Recursing on remaining parameters", param_settings_left
					self.recurse_and_append_params(copy.deepcopy(param_settings_left), 
												   copy.deepcopy(this_param_dict))
				if debug: print "Placing the popped setting", key, "=", this_setting, "into the parameter dict"
				this_param_dict[key] = this_setting
				
		self.list_of_param_settings.append(this_param_dict)
		if debug: print "Appending parameter dict to list:", this_param_dict, "\n"

	def load_data(self):
		self.test_tasks = helper.loadPickledTaskList(self.datasets_path, self.file_prefix, "Test",fix_y=True)
		self.train_tasks = helper.loadPickledTaskList(self.datasets_path, self.file_prefix, "Train",fix_y=True)
		self.n_tasks = len(self.train_tasks)
	
	def get_save_prefix(self, file_prefix, replace=False):
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

		prefix = self.classifier_name + task_str + file_prefix[dash_loc:-1] + name_modifier

		if not replace:
			while os.path.exists(self.results_path + prefix + '.csv'):
				prefix = prefix + '2'
		return prefix

	def setting_already_done(self, param_dict):
		mini_df = self.val_results_df
		for key in param_dict.keys():
			mini_df = mini_df[mini_df[key] == param_dict[key]]
			if len(mini_df) == 0:
				return False
		print "Setting already tested"
		return True

	def convert_param_dict_for_use(self, param_dict):
		"""When loading rows from a saved results df in csv format, some 
		of the settings may end up being converted to a string representation
		and need to be converted back to actual numbers and objects.
		
		May need to be overwritten in child class.""" 
		param_dict['task_num'] = int(param_dict['task_num'])
		return param_dict

	def get_preds_true_for_task(self,train_tasks, test_tasks, param_dict):
		t = param_dict['task_num']
		X = train_tasks[t]['X']
		y = train_tasks[t]['Y']

		test_X = test_tasks[t]['X']
		true_y = list(test_tasks[t]['Y'].flatten())

		if len(y)==0 or len(X)==0 or len(test_X) == 0 or len(true_y)==0:
			return None, None

		if self.cant_train_with_one_class and len(np.unique(y))==1:
			preds = list(np.unique(y)[0]*np.ones(len(true_y)))
		else:
			preds = self.train_and_predict_task(t, X, y, test_X, param_dict)

		return preds, true_y

	def sweep_all_parameters(self):
		print "\nYou have chosen to test a total of", self.num_settings / self.n_tasks, "settings"
		print "for each of", self.n_tasks, "tasks, leading to a total of..."
		print self.num_settings, "models to train!!"
		sys.stdout.flush()

		#sweep all possible combinations of parameters
		for t in range(self.n_tasks):
			print "\nSweeping all parameters for task t:", self.train_tasks[t]['Name']
			for param_dict in self.list_of_param_settings:
				these_params = copy.deepcopy(param_dict)
				these_params['task_num'] = t
				these_params['task_name'] = self.train_tasks[t]['Name']
				self.test_one_setting(these_params)
			
			self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

	def test_one_setting(self, param_dict):
		if self.cont and self.setting_already_done(param_dict):
			return
		t0 = time()
		
		results_dict = self.get_cross_validation_results(param_dict)
		self.val_results_df = self.val_results_df.append(results_dict,ignore_index=True)
		
		t1 = time()
		this_time = t1 - t0
		self.time_sum = self.time_sum + this_time
		
		print "\n", self.val_results_df.tail(n=1)
		print "It took", this_time, "seconds to obtain this result"
		self.print_time_estimate()
		
		sys.stdout.flush()

		#output the file every few iterations for safekeeping 
		if len(self.val_results_df) % self.save_results_every_nth == 0:
			self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

	def get_cross_validation_results(self, param_dict, print_per_fold=False):
		all_acc = []
		all_auc = []
		all_f1 = []
		all_precision = []
		all_recall = []

		for f in range(self.num_cross_folds):
			train_tasks, val_tasks = helper.loadCrossValData(self.datasets_path, self.file_prefix, f, fix_y=True)		
			
			preds, true_y = self.get_preds_true_for_task(train_tasks, val_tasks, param_dict)
			if preds is None or true_y is None:
				continue

			acc, auc, f1, precision, recall = helper.computeAllMetricsForPreds(preds, true_y)
			all_acc.append(acc)
			all_auc.append(auc)
			all_f1.append(f1)
			all_precision.append(precision)
			all_recall.append(recall)
			if print_per_fold: print "Fold", f, "acc", acc, "auc", auc, "f1", f1, "precision",precision,"recall",recall

		if print_per_fold:
			print "accs for all folds", all_acc
			print "aucs for all folds", all_auc
		
		# Add results to the dictionary
		param_dict['val_acc'] = np.nanmean(all_acc)
		param_dict['val_auc'] = np.nanmean(all_auc)
		param_dict['val_f1'] = np.nanmean(all_f1)
		param_dict['val_precision'] = np.nanmean(all_precision)
		param_dict['val_recall'] = np.nanmean(all_recall)

		return param_dict

	def print_time_estimate(self):
		num_done = len(self.val_results_df)-self.started_from
		num_remaining = self.num_settings - num_done - self.started_from
		avg_time = self.time_sum / num_done
		total_secs_remaining = int(avg_time * num_remaining)
		hours = total_secs_remaining / 60 / 60
		mins = (total_secs_remaining % 3600) / 60
		secs = (total_secs_remaining % 3600) % 60

		print "\n", num_done, "settings processed so far,", num_remaining, "left to go"
		print "Estimated time remaining:", hours, "hours", mins, "mins", secs, "secs"

	def get_baseline(self, Y):
		Y = Y.tolist()
		percent_true = float(Y.count(1.0)) / float(len(Y))
		if percent_true < 0.5:
			return 1.0 - percent_true
		else:
			return percent_true

	def find_best_setting_for_task(self, task_num, optimize_for='val_acc'):
		task_df = self.val_results_df[self.val_results_df['task_num']==task_num]
		accuracies = task_df[optimize_for].tolist()
		max_acc = max(accuracies)
		max_idx = accuracies.index(max_acc)
		return task_df.iloc[max_idx]

	def get_final_results(self, optimize_for='val_acc'):
		if self.users_as_tasks and not self.check_test:
			print "check_test is set to false, Will not evaluate performance on held-out test set."
			return
		print "\nAbout to evaluate results on held-out test set!!"
		print "Will use the settings that produced the best", optimize_for
		
		all_preds = []
		all_true_y = []
		per_task_accs = []
		per_task_aucs = []
		per_task_f1 = []
		per_task_precision = []
		per_task_recall = []

		for t in range(self.n_tasks):
			task_settings = self.find_best_setting_for_task(t, optimize_for=optimize_for)
			assert(task_settings['task_num'] == t)
			if not self.users_as_tasks:
				print "\nBEST SETTING FOR TASK", t, "-", task_settings['task_name']
				print "The highest", optimize_for, "of", task_settings[optimize_for], "was found with the following settings:"
				print task_settings

			task_settings = self.convert_param_dict_for_use(task_settings)
			preds, true_y = self.get_preds_true_for_task(self.train_tasks, self.test_tasks, task_settings)
			if preds is None or true_y is None:
				continue

			all_preds.extend(preds)
			all_true_y.extend(true_y)

			# save the per-task results
			t_acc, t_auc, t_f1, t_precision, t_recall = helper.computeAllMetricsForPreds(preds, true_y)
			per_task_accs.append(t_acc)
			per_task_aucs.append(t_auc)
			per_task_f1.append(t_f1)
			per_task_precision.append(t_precision)
			per_task_recall.append(t_recall)

			if not self.users_as_tasks:
				print "\nFINAL TEST RESULTS FOR", helper.getFriendlyLabelName(self.train_tasks[t]['Name'])
				print 'Acc:', t_acc, 'AUC:', t_auc, 'F1:', t_f1, 'Precision:', t_precision, 'Recall:', t_recall

		print "\nHELD OUT TEST METRICS COMPUTED BY AVERAGING OVER TASKS"
		avg_acc = np.nanmean(per_task_accs)
		avg_auc = np.nanmean(per_task_aucs)
		avg_f1 = np.nanmean(per_task_f1)
		avg_precision = np.nanmean(per_task_precision)
		avg_recall = np.nanmean(per_task_recall)
		print 'Acc:', avg_acc, 'AUC:', avg_auc, 'F1:', avg_f1, 'Precision:', avg_precision, 'Recall:', avg_recall

		if self.test_csv_filename is not None:
			print "\tSAVING HELD OUT PREDICITONS"
			if self.users_as_tasks:
				task_column = 'user_id'
				label_name = helper.getFriendlyLabelName(self.file_prefix)
				wanted_label = helper.getOfficialLabelName(label_name)
				predictions_df = helper.get_test_predictions_for_df_with_task_column(
						self.predict_task, self.test_csv_filename, task_column, self.test_tasks, 
						wanted_label=wanted_label, num_feats_expected=np.shape(self.test_tasks[0]['X'])[1], 
						label_name=label_name, tasks_are_ints=False)
			else:
				predictions_df = helper.get_test_predictions_for_df_with_no_task_column(self.predict_task, 
					self.test_csv_filename, self.test_tasks, num_feats_expected=np.shape(self.test_tasks[0]['X'])[1])
			predictions_df.to_csv(self.results_path + "Preds-" + self.save_prefix + '.csv')
		else:
			print "Uh oh, the test csv filename was not set, can't save test preds"

	def run(self):
		self.sweep_all_parameters()
		self.get_final_results()

