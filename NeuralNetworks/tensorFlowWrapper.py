import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import os
import pickle
from time import time

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_RESULTS_PATH = '/Your/path/here/'
DEFAULT_DATASETS_PATH = '/Your/path/here/'
DEFAULT_FIGURES_PATH = '/Your/path/here/'

DEFAULT_VAL_TYPE = 'cross'
OUTPUT_EVERY_NTH = 3

sys.path.append(PATH_TO_REPO)
import tensorFlowNetwork as tfnet
import tensorFlowNetworkMultiTask as mtltf
import helperFuncs as helper

def reloadFiles():
	reload(tfnet)
	reload(mtltf)
	reload(helper)
	tfnet.reloadHelper()
	mtltf.reloadFiles()

class TensorFlowWrapper:
	def __init__(self, dataset_name, target_label=None, trial_name=None, multilabel=False, multitask=False, 
				 print_per_task=False, test_steps=9001, results_path=DEFAULT_RESULTS_PATH, 
				 datasets_path=DEFAULT_DATASETS_PATH, figures_path=DEFAULT_FIGURES_PATH, val_output_file=None, 
				 val_type=DEFAULT_VAL_TYPE, cont=False, architectures=None, test_csv_filename=None):
		assert not(multilabel and multitask)

		self.multilabel = multilabel
		self.multitask = multitask
		self.results_path = results_path
		self.figures_path = figures_path
		self.datasets_path = datasets_path
		self.dataset_name = dataset_name 
		self.test_steps = test_steps
		self.val_type = val_type
		self.cont = cont
		self.print_per_task = print_per_task
		if test_csv_filename is not None:
			self.test_csv_filename = self.datasets_path + test_csv_filename
		else:
			self.test_csv_filename = None
		if cont:
			replace = True
		else:
			replace = False
		if trial_name is None and target_label is not None:
			trial_name = helper.getFriendlyLabelName(target_label)
		self.trial_name = trial_name
		self.val_output_prefix = self.getValOutputName(val_output_file, dataset_name, trial_name, replace=replace)  

		#dataset stuff
		if multitask:
			train_tasks = pickle.load(open(self.datasets_path + dataset_name + "Train.p","rb"))
			val_tasks = pickle.load(open(self.datasets_path + dataset_name + "Val.p","rb"))
			test_tasks = pickle.load(open(self.datasets_path + dataset_name + "Test.p","rb"))

			self.net = mtltf.TensorFlowNetworkMTL(train_tasks, val_tasks, test_tasks, verbose=False, 
												  val_type=self.val_type, print_per_task=print_per_task)
			self.wanted_labels = self.net.optimize_labels
		else:
			self.data_df = pd.DataFrame.from_csv(self.datasets_path + self.dataset_name)
			self.wanted_feats = [x for x in self.data_df.columns.values if x != 'user_id' and x != 'timestamp' and x!= 'dataset' and '_Label' not in x]
			if self.multilabel:
				self.wanted_labels = [x for x in self.data_df.columns.values if '_Label' in x and 'tomorrow_' in x and 'Evening' in x and 'Alertness' not in x and 'Energy' not in x]
				self.optimize_labels = [x for x in self.wanted_labels if 'tomorrow_' in x and 'Evening_' in x]
			else:
				self.wanted_labels = [target_label]

			#actual network
			self.net = tfnet.TensorFlowNetwork(self.data_df, self.wanted_feats, self.wanted_labels, optimize_labels=self.wanted_labels,
											multilabel=self.multilabel, verbose=False, val_type=self.val_type)

		#parameters that can be tuned:
		self.l2_regularizers = [1e-2, 1e-4]
		self.dropout = [True, False]
		self.decay = [True]
		self.decay_steps = [1000]
		self.decay_rates = [0.95]
		self.optimizers = [tf.train.AdamOptimizer] #[tf.train.AdagradOptimizer,  tf.train.GradientDescentOptimizer
		self.train_steps =[5001]
		if multitask:
			self.batch_sizes = [20]
			self.learning_rates = [.01, .001, .0001]
			self.architectures = [[500,50],[300,20,10]] if architectures is None else architectures
		else:
			self.batch_sizes = [50,75]
			self.learning_rates = [.01, .001, .0001]
			self.architectures = [[1024,256],[500,50],[1024]] if architectures is None else architectures

		#storing the results
		self.time_sum = 0
		if cont:
			self.val_results_df = pd.DataFrame.from_csv(self.results_path + self.val_output_prefix + '.csv')
			print '\nPrevious validation results df loaded. It has', len(self.val_results_df), "rows"
			self.started_from = len(self.val_results_df)
		else:
			self.val_results_df = pd.DataFrame()
			self.started_from = 0

	def getValOutputName(self, val_output_file, dataset_file, trial_name, replace=False):
		if self.multitask:
			multilabel_str = 'MTL_'
		elif self.multilabel:
			multilabel_str = 'multilabel_'
		else:
			multilabel_str = ''

		name_modifier = ""
		if '/' in dataset_file:
			if "NoLocation" in dataset_file:
				name_modifier = "-noloc"
			slash_loc = dataset_file.find('/')
			dataset_file = dataset_file[slash_loc+1:]

		if replace or val_output_file is None:
			val_output_file = 'nn_' + multilabel_str + dataset_file[0:-4] + name_modifier + "_"
		if trial_name is not None:
			val_output_file = val_output_file + trial_name
		if not replace:
			while os.path.exists(self.results_path + val_output_file + '.csv') \
				or os.path.exists(self.figures_path + val_output_file + '.eps'):
				val_output_file = val_output_file + '2'
		return val_output_file

	def setNetworkArchitecturesToTest(self, architectures):
		self.architectures = architectures

	def constructNetwork(self, hidden_layers):
		if self.multitask:
			hidden_layers_shared = hidden_layers[:-1]
			hidden_task_nodes = hidden_layers[-1]
			connections_shared = ['full'] * (len(hidden_layers))
			self.net.setUpNetworkStructure(hidden_layers_shared,hidden_task_nodes,connections_shared,['full','full'])
		else:
			connections = ['full'] * (len(hidden_layers)+1)
			self.net.setUpNetworkStructure(hidden_layers,connections)

	# use something like the following to test only one set of parameters:
	# wrapper.setParams(l2_regularizers=[1e-4], learning_rates=[.01], dropout=[True], decay=[True], batch_sizes=[50], optimizers=[tf.train.GradientDescentOptimizer])
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

	def settingAlreadyDone(self, hidden_layers, l2_beta, lrate, dropout, decay, dsteps, drate, bsize, opt, tsteps):
		if len(self.val_results_df[(self.val_results_df['hidden_layers']== str(hidden_layers)) & \
									(self.val_results_df['l2_beta']== l2_beta) & \
									(self.val_results_df['learning_rate']== lrate) & \
									(self.val_results_df['dropout']== dropout) & \
									(self.val_results_df['decay']== decay) & \
									(self.val_results_df['decay_steps']== dsteps) & \
									(self.val_results_df['decay_rate']== drate) & \
									(self.val_results_df['batch_size']== bsize) & \
									(self.val_results_df['optimizer']== str(opt))]) > 0:
			print "setting already tested"
			return True
		else:
			return False

	def testOneSetting(self, hidden_layers, l2_beta, lrate, dropout, decay, dsteps, drate, bsize, opt, tsteps, num_settings):
		print "Testing setting with layers", hidden_layers, "beta", l2_beta, "lrate", lrate, "dropout", dropout, "decay", decay, "dsteps", dsteps, "drate", drate, "bsize", bsize, "opt", opt, "tsteps", tsteps
		if self.cont:
			if self.settingAlreadyDone(hidden_layers, l2_beta, lrate, dropout, decay, dsteps, drate, bsize, opt, tsteps):
				return

		t0 = time()
		self.net.setParams(l2_beta=l2_beta, initial_learning_rate=lrate, decay=decay, 
							decay_steps=dsteps, decay_rate=drate, batch_size=bsize,
							optimizer=opt, n_steps=tsteps, dropout=dropout)
		self.constructNetwork(hidden_layers)
		if self.val_type == 'cross':
			acc, auc, f1, precision, recall = self.net.trainAndCrossValidate()
		else:
			acc, auc, f1, precision, recall = self.net.trainAndValidate()

		results_dict = {'hidden_layers':hidden_layers, 'l2_beta': l2_beta, 'learning_rate': lrate, 
						'dropout': dropout, 'decay': decay, 'decay_steps': dsteps, 
						'decay_rate': drate, 'batch_size': bsize, 
						'optimizer': opt, 'val_acc': acc, 'val_auc':auc,
						'val_f1':f1, 'val_precision':precision, 'val_recall':recall}
		if self.multitask:
			results_dict['train_nan_percent'] = self.net.train_nan_percent[-1]
			results_dict['val_nan_percent'] = self.net.val_nan_percent[-1]

		if self.multilabel or self.print_per_task:
			for label in self.wanted_labels:
				friendly_label = helper.getFriendlyLabelName(label)
				results_dict[friendly_label + '_acc'] = self.net.training_val_results_per_task['acc'][label][-1]
				results_dict[friendly_label + '_auc'] = self.net.training_val_results_per_task['auc'][label][-1]
				results_dict[friendly_label + '_f1'] = self.net.training_val_results_per_task['f1'][label][-1]
				results_dict[friendly_label + '_precision'] = self.net.training_val_results_per_task['precision'][label][-1]
				results_dict[friendly_label + '_recall'] = self.net.training_val_results_per_task['recall'][label][-1]
		self.val_results_df = self.val_results_df.append(results_dict,ignore_index=True)
		
		print self.val_results_df.tail(n=1)
		t1 = time()
		this_time = t1 - t0
		print "It took", this_time, "seconds to obtain this result"

		self.time_sum = self.time_sum + this_time

		self.printTimeEstimate(len(self.val_results_df)-self.started_from, num_settings)
		sys.stdout.flush()

		#output the file every few iterations for safekeeping 
		if len(self.val_results_df) % OUTPUT_EVERY_NTH == 0:
			self.val_results_df.to_csv(self.results_path + self.val_output_prefix + '.csv')

	def printTimeEstimate(self, num_done, num_desired):
		num_remaining = num_desired - num_done
		avg_time = self.time_sum / num_done
		total_secs_remaining = int(avg_time * num_remaining)
		hours = total_secs_remaining / 60 / 60
		mins = (total_secs_remaining % 3600) / 60
		secs = (total_secs_remaining % 3600) % 60

		print "\n", num_done, "settings processed so far,", num_remaining, "left to go"
		print "Estimated time remaining:", hours, "hours", mins, "mins", secs, "secs"

	def calcNumSettingsPerStructure(self):
		num_settings = len(self.l2_regularizers) * len(self.learning_rates) * len(self.dropout) * len(self.decay) \
						* len(self.batch_sizes)  * len(self.optimizers) * len(self.train_steps)
		if True in self.decay and (len(self.decay_steps) > 1 or len(self.decay_rates) > 1):
			num_settings = num_settings * ((len(self.decay_steps) * len(self.decay_rates)) / 2.0) 
		return num_settings

	def sweepParameters(self, hidden_layers, num_settings):
		print "\nSweeping all parameters for structure:", hidden_layers
	
		#sweep all possible combinations of parameters
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
												self.testOneSetting(hidden_layers, l2_beta, lrate, dropout, decay, dsteps, drate, bsize, opt, tsteps, num_settings)
									else:
										#decay steps and decay rate don't matter if decay is set to false
										self.testOneSetting(hidden_layers, l2_beta, lrate, dropout, decay, 10000, 0.95, bsize, opt, tsteps, num_settings)
		self.val_results_df.to_csv(self.results_path + self.val_output_prefix + '.csv')

	def sweepStructuresAndParameters(self):
		num_settings = self.calcNumSettingsPerStructure()
		num_settings_total = num_settings * len(self.architectures)

		print "\nYou have chosen to test", num_settings, "settings for each of", len(self.architectures), "architectures"
		print "This is a total of", num_settings_total, "tests."
		for hidden_layers in self.architectures:
			self.sweepParameters(hidden_layers,num_settings_total)

	def findBestSetting(self, retrain_and_plot=True, optimize_for='val_auc'):
		accuracies = self.val_results_df[optimize_for].tolist()
		max_acc = max(accuracies)
		max_idx = accuracies.index(max_acc)
		best_setting = self.val_results_df.iloc[max_idx]

		print "BEST SETTING!"
		print "The highest", optimize_for, "of", max_acc, "was found with the following settings:"
		print best_setting

		best_setting = helper.fixSettingDictLoadedFromResultsDf(best_setting)

		if retrain_and_plot:
			self.retrainAndPlot(best_setting)
		else:
			return best_setting

	def retrainAndPlot(self, setting_dict):
		print "\nRETRAINING WITH THE BEST SETTINGS:"

		self.net.verbose = True
		self.net.setParams(l2_beta=setting_dict['l2_beta'], initial_learning_rate=setting_dict['learning_rate'], decay=setting_dict['decay'], 
							decay_steps=setting_dict['decay_steps'], decay_rate=setting_dict['decay_rate'], batch_size=setting_dict['batch_size'],
							optimizer=setting_dict['optimizer'], dropout=setting_dict['dropout'])
		self.constructNetwork(setting_dict['hidden_layers'])

		self.net.setUpGraph()
		self.net.runGraph(self.test_steps, print_test=True)

		if self.multilabel:
			for label in self.optimize_labels:
				friendly_label = helper.getFriendlyLabelName(label)
				self.net.plotValResults(save_path=self.figures_path + self.val_output_prefix + '-' + friendly_label + '.eps', label=label)
				self.net.plotValResults(save_path=self.figures_path + self.val_output_prefix + '-' + friendly_label + '.png', label=label)
				print "Final validation results for", friendly_label,"... Acc:", \
						self.net.training_val_results_per_task['acc'][label][-1], "Auc:", self.net.training_val_results_per_task['auc'][label][-1]
		elif self.print_per_task:
			for label in self.wanted_labels:
				friendly_label = helper.getFriendlyLabelName(label)
				self.net.plotValResults(save_path=self.figures_path + self.val_output_prefix + '-' + friendly_label + '.eps', label=label)
				self.net.plotValResults(save_path=self.figures_path + self.val_output_prefix + '-' + friendly_label + '.png', label=label)
				print "Final validation results for", friendly_label,"... Acc:", \
					self.net.training_val_results_per_task['acc'][label][-1], "Auc:", self.net.training_val_results_per_task['auc'][label][-1]
		else:
			self.net.plotValResults(save_path=self.figures_path + self.val_output_prefix + '.eps')
			self.net.plotValResults(save_path=self.figures_path + self.val_output_prefix + '.png')
			print "Final AUC:", self.net.training_val_results['auc'][-1]

		if self.test_csv_filename is not None:
			if self.multitask:
				task_column = None
				if 'Cluster' in self.dataset_name:
					print "Guessing the task column is Big5GenderKMeansCluster - if this is incorrect expect errors"
					task_column = 'Big5GenderKMeansCluster'
					tasks_are_ints = True
				
				if 'User' in self.dataset_name:
					print "Guessing the task column is user_id - if this is incorrect expect errors"
					task_column = 'user_id'
					tasks_are_ints = False
				
				if task_column is not None:
					label_name = helper.getFriendlyLabelName(self.dataset_name)
					wanted_label = helper.getOfficialLabelName(label_name)
					test_preds_df = helper.get_test_predictions_for_df_with_task_column(
						self.net.predict, self.test_csv_filename, task_column, self.net.test_tasks, 
						wanted_label=wanted_label, num_feats_expected=np.shape(self.net.test_tasks[0]['X'])[1], 
						label_name=label_name, tasks_are_ints=tasks_are_ints)
				else:
					test_preds_df = helper.get_test_predictions_for_df_with_no_task_column(self.net.predict, self.test_csv_filename,
																	self.net.test_tasks, 
																	num_feats_expected=np.shape(self.net.test_tasks[0]['X'])[1])
			else:
				test_preds_df = self.net.get_preds_for_df()
			print "Got a test preds df! Saving it to:", self.results_path + "Preds-" + self.val_output_prefix + '.csv'
			test_preds_df.to_csv(self.results_path + 'Preds-' + self.val_output_prefix + '.csv')
		else:
			print "Uh oh, the test csv filename was not set, can't save test preds"

		print "Saving a copy of the final model!"
		self.net.save_model(self.val_output_prefix, self.results_path)
		

	def run(self):
		self.sweepStructuresAndParameters()
		self.findBestSetting()

if __name__ == "__main__":
	print "TENSOR FLOW MODEL SELECTION"
	print "\tThis code will sweep a set of network architectures and parameters to find the ideal settings for a single dataset"

	if len(sys.argv) < 4:
		print "Error: usage is python tensorFlowWrapper.py <data file> <classification type> <multitasking over> <continue>"
		print "\t<data file>: e.g. dataset-Simple-Group.csv or datasetTaskList-Discard40-Future-Personal_ ... Program will look in the following directory for this file", DEFAULT_DATASETS_PATH
		print "\t<classification type>:"
		print "\t\tFor single task learning, enter the name of the label you would like classify on. E.g. Group_Happiness_Evening_Label"
		print "\t\tFor multi task learning, in which the same net learns several tasks (like several wellbeing measures) enter: multilabel"
		print "\t\tFor multi task learning, in which each task gets its own piece of the network, but the first layers are shared (like users as tasks) enter: multitask"
		print "\t<multitasking over> For wellbeing-ask-tasks use 'wellbeing', for users-as-tasks use 'users'"
		print "\t<continue>: optional. If 'True', the neural net will pick up from where it left off by loading a previous validation results file"
		print "\t<csv file for testing>: optional. If you want to get the final test results, provide the name of a csv file to test on"
		sys.exit()
	filename= sys.argv[1] #get data file from command line argument
	print "\nLoading dataset", DEFAULT_DATASETS_PATH + filename
	print ""

	multilabel = False
	multitask = False
	target_label = None
	if sys.argv[2] == 'multilabel':
		multilabel = True
		print "Performing multi-task classification, in which the same net is shared by all tasks"
		print "Optimizing for accuracy on tomorrow evening"
	elif sys.argv[2] == 'multitask':
		multitask = True
		print "Performing multi-task classification, in which each task gets it's own private final hidden layer"
	else:
		target_label = sys.argv[2]
		print "Performing single-task classification, classifying on", target_label

	if sys.argv[3] == 'wellbeing':
		print_per_task = True
	else:
		print_per_task = False

	if len(sys.argv) >= 5 and sys.argv[4] == 'True':
		cont = True
		print "Okay, will continue from a previously saved validation results file for this problem"
	else:
		cont = False
	print ""

	if len(sys.argv) >= 6:
		csv_test_file = sys.argv[5]
		print "Okay, will get final test results on file", csv_test_file
		print ""
	else:
		csv_test_file = None

	wrapper = TensorFlowWrapper(filename, target_label=target_label, multilabel=multilabel, multitask=multitask, 
							    print_per_task=print_per_task, cont=cont, test_csv_filename=csv_test_file)
	
	print "\nThe following parameter settings will be tested:"
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

	print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.val_output_prefix + '.csv'
	print "\nThe validation accuracy figures will be saved in:", wrapper.figures_path + wrapper.val_output_prefix + '.eps'

	wrapper.run()