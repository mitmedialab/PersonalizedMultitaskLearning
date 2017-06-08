
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pickle
import sys
import os
import copy
from time import time
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_RESULTS_PATH = '/Your/path/here/'
DEFAULT_DATASETS_PATH = '/Your/path/here/'
DEFAULT_FIGURES_PATH = '/Your/path/here/'

import HBLR as hblr
import helperFuncs as helper

DEFAULT_NUM_CROSS_FOLDS = 5
DEFAULT_MAX_ITERS = 75
SAVE_RESULTS_EVERY_X_TESTS = 1
DEFAULT_VALIDATION_TYPE = 'cross'


'''	Notes:
	-Parameters to tune: tau10, tau20, mu, sigma
		-ratio between tau10 and tau20 controls the number of clusters. A greater ratio = more clusters
			-successful run was done with tau10 = tau20 = 0.05
		-small sigma might be good. e.g. 0.1*I
		-mu is usually 0. not testing for now
	-set number of clusters:
		-for wellbeing measures as tasks go with default (K=num_tasks)
		-for users as tasks no more than 25 
'''

def reloadFiles():
	reload(hblr)
	reload(helper)

class HBLRWrapper:

	def __init__(self, file_prefix, users_as_tasks=False, num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, cont=False,
				results_path=DEFAULT_RESULTS_PATH, figures_path=DEFAULT_FIGURES_PATH, datasets_path=DEFAULT_DATASETS_PATH, 
				test_run=False, max_iters=DEFAULT_MAX_ITERS, val_type=DEFAULT_VALIDATION_TYPE, optimize_labels=None,
				test_csv_filename=None):
		self.results_path = results_path
		self.figures_path = figures_path
		self.datasets_path = datasets_path
		self.save_prefix = self.getSavePrefix(file_prefix, replace=cont)
		self.cont=cont
		self.max_iters = max_iters
		self.val_type = val_type
		self.users_as_tasks = users_as_tasks
		self.file_prefix = file_prefix
		if test_csv_filename is not None:
			self.test_csv_filename = self.datasets_path + test_csv_filename
		else:
			self.test_csv_filename = None
		self.test_tasks = helper.loadPickledTaskList(datasets_path, file_prefix, "Test")
		self.train_tasks = helper.loadPickledTaskList(datasets_path, file_prefix, "Train")
		if self.val_type != 'cross':
			self.val_tasks = helper.loadPickledTaskList(datasets_path, file_prefix, "Val")
			self.initializeHBLRModel(self.train_tasks)
		else:
			self.classifier = None
		
		if users_as_tasks:
			self.K = 25
		else:
			self.K = len(self.test_tasks)
		self.n_feats = helper.calculateNumFeatsInTaskList(self.test_tasks)
		self.n_tasks = len(self.test_tasks)

		if optimize_labels is None:
			self.optimize_labels = ['tomorrow_Group_Happiness_Evening_Label', 'tomorrow_Group_Health_Evening_Label', 'tomorrow_Group_Calmness_Evening_Label']
		else:
			self.optimize_labels = optimize_labels

		#parameters that can be tuned
		self.tau10s=[10, 1, 0.05, 0.01]
		self.tau20s=[1.0, 0.05, 0.01]
		self.sigma_multipliers = [.01,0.1, 1]
		self.mu_multipliers = [0.0]

		if test_run:
			print "This is only a testing run. Using cheap settings to make it faster"
			self.K = 2
			self.max_iters = 5
			self.n_tasks = 2
			self.tau10s=[1]
			self.tau20s=[.1]
			self.sigma_multipliers=[.01]
			self.mu_multipliers=[0]

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


	def initializeHBLRModel(self, train_tasks):
		self.classifier = hblr.HBLR(train_tasks,K=self.K, debug=False,max_iterations=self.max_iters, verbose=False)

	def getSavePrefix(self, file_prefix, replace=False):
		dash_loc = file_prefix.find('-')
		prefix = "hblr" + file_prefix[dash_loc:-1]
		if not replace:
			while os.path.exists(self.results_path + prefix + '.csv'):
				prefix = prefix + '2'
		return prefix

	def calcNumSettingsDesired(self):
		self.num_settings = len(self.tau10s) * len(self.tau20s) * len(self.mu_multipliers)  \
						* len(self.sigma_multipliers)

	# use something like the following to test only one set of parameters:
	# wrapper.setParams(tau10s=[.05], tau20s=[.05], sigma_multipliers=[.1,.01])
	def setParams(self, tau10s=None, tau20s=None, sigma_multipliers=None, mu_multipliers=None):
		'''does not override existing parameter settings if the parameter is not set'''
		self.tau10s = tau10s if tau10s is not None else self.tau10s
		self.tau20s = tau20s if tau20s is not None else self.tau10s
		self.sigma_multipliers = sigma_multipliers if sigma_multipliers is not None else self.sigma_multipliers
		self.mu_multipliers = mu_multipliers if mu_multipliers is not None else self.mu_multipliers

	def settingAlreadyDone(self, tau10, tau20, mu_mult, sigma_mult):
		if len(self.val_results_df[(self.val_results_df['tau10']== tau10) & \
									(self.val_results_df['tau20']== tau20) & \
									(self.val_results_df['sigma_multiplier']== mu_mult) & \
									(self.val_results_df['mu_multiplier']== sigma_mult)]) > 0:
			print "setting already tested"
			return True
		else:
			return False

	def setClassifierToSetting(self, tau10, tau20, sigma_mult, mu_mult):
		sigma = sigma_mult * np.eye(self.n_feats)
		mu = mu_mult * np.ones((1,self.n_feats))

		self.classifier.setHyperParameters(mu, sigma, tau10, tau20)

	def getAccuracyAucOnAllTasks(self, task_list):
		all_task_Y = []
		all_preds = []
		for i in range(len(task_list)):
			preds, task_Y = self.getPredsTrueOnOneTask(task_list,i)
			if preds is None:
				# Skipping task because it does not have valid data
				continue
			if len(task_Y)>0:
				all_task_Y.extend(task_Y)
				all_preds.extend(preds)
		if not helper.containsEachLabelType(all_preds):
			print "for some bizarre reason, the preds for all tasks are the same class"
			print "preds", all_preds
			print "true_y", all_task_Y
			auc = np.nan
		else:
			auc=roc_auc_score(all_task_Y, all_preds)
		acc=hblr.getBinaryAccuracy(all_preds,all_task_Y)
		return acc,auc

	def getPredsTrueOnOneTask(self, task_list, task):
		if not helper.isValidTask(task_list, task):
			return None, None
		task_Y = list(task_list[task]["Y"])
		return self.classifier.predictBinary(task_list[task]['X'], task), task_Y
		
	def getAccuracyAucOnOneTask(self, task_list, task):
		preds, task_Y = self.getPredsTrueOnOneTask(task_list,task)
		if preds is None:
			# Returning nan for task because it does not have valid data
			return np.nan, np.nan
		acc = hblr.getBinaryAccuracy(preds,task_Y)
		if len(task_Y) <= 1 or not helper.containsEachLabelType(preds):
			auc = np.nan
		else:
			auc = roc_auc_score(task_Y, preds)
		return acc,auc

	def getValidationResults(self, results_dict):
		self.classifier.trainUntilConverged()
		results_dict['num_clusters'] = self.classifier.K

		if self.users_as_tasks:
			val_acc, val_auc = self.getAccuracyAucOnAllTasks(self.val_tasks)
			results_dict['val_acc'] = val_acc
			results_dict['val_auc'] = val_auc
		else:
			accs = []
			aucs = []
			for t in range(self.n_tasks):
				acc, auc = self.getAccuracyAucOnOneTask(self.val_tasks, t)
				task_name = self.val_tasks[t]['Name']
				results_dict['TaskAcc-' + helper.getFriendlyLabelName(task_name)] = acc
				results_dict['TaskAuc-' + helper.getFriendlyLabelName(task_name)] = auc
				if task_name in self.optimize_labels:
					accs.append(acc)
					aucs.append(auc)
			results_dict['val_acc'] = np.nanmean(accs)
			results_dict['val_auc'] = np.nanmean(aucs)
		return results_dict

	def getCrossValidationResults(self, results_dict, tau10, tau20, sigma_mult, mu_mult, save_plots=False, print_per_fold=False):
		if save_plots:
			same_task_matrix = np.zeros((self.n_tasks,self.n_tasks))

		clusters = [0] * self.num_cross_folds

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
			train_tasks, val_tasks = helper.loadCrossValData(self.datasets_path, self.file_prefix, f, reshape=True)

			self.initializeHBLRModel(train_tasks)
			self.setClassifierToSetting(tau10, tau20, sigma_mult, mu_mult)
			self.classifier.trainUntilConverged()

			clusters[f] = self.classifier.K
		
			if save_plots: same_task_matrix = self.updateSameTaskMatrix(same_task_matrix)

			# Get results!
			fold_preds = []
			fold_true_y = []
			for t in range(self.n_tasks):
				preds = self.classifier.predictBinary(val_tasks[t]['X'], t)
				true_y = list(val_tasks[t]['Y'].flatten())

				if len(preds)==0 or len(true_y) == 0:
					continue

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
		print "clusters for all folds", clusters

		if save_plots:
			self.plotAccuracyAucAndClusters(all_acc, all_auc, clusters)
			self.saveHintonPlot(same_task_matrix, self.num_cross_folds) 
			pd.DataFrame(same_task_matrix).to_csv(self.results_path + self.save_prefix + "-same_task_matrix.csv")
		
		# Add results to the dictionary
		results_dict['val_acc'] = np.nanmean(all_acc)
		results_dict['val_auc'] = np.nanmean(all_auc)
		results_dict['val_f1'] = np.nanmean(all_f1)
		results_dict['val_precision'] = np.nanmean(all_precision)
		results_dict['val_recall'] = np.nanmean(all_recall)
		results_dict['num_clusters'] = np.nanmean(clusters)

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

	def testOneSetting(self, tau10, tau20, sigma_mult, mu_mult):
		if self.cont:
			if self.settingAlreadyDone(tau10, tau20, sigma_mult, mu_mult):
				return

		t0 = time()
		
		results_dict = {'tau10':tau10, 'tau20': tau20, 'sigma_multiplier':sigma_mult, 'mu_multiplier':mu_mult}
		
		if self.val_type == 'cross':
			results_dict = self.getCrossValidationResults(results_dict, tau10, tau20, sigma_mult, mu_mult)
		else:
			self.setClassifierToSetting(tau10, tau20, sigma_mult, mu_mult)
			results_dict = self.getValidationResults(results_dict)
		
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
		for tau10 in self.tau10s:
			for tau20 in self.tau20s:
				for sigma_mult in self.sigma_multipliers:
					for mu_mult in self.mu_multipliers:
						self.testOneSetting(tau10, tau20, sigma_mult, mu_mult)
		self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

	def findBestSetting(self, save_final_results=False):
		accuracies = self.val_results_df['val_acc'].tolist()
		max_acc = max(accuracies)
		max_idx = accuracies.index(max_acc)

		print "BEST SETTING!"
		print "The highest validation accuracy of", max_acc, "was found with the following settings:"
		print self.val_results_df.iloc[max_idx]

		if self.test_csv_filename is not None or save_final_results:
			self.getFinalResultsAndSave(self.val_results_df.iloc[max_idx])
		else:
			print "Not running Final results"
			return self.val_results_df.iloc[max_idx]

	def run(self):
		self.sweepAllParameters()
		return self.findBestSetting()

	def getFinalResultsAndSave(self, setting_dict):
		if self.val_type == 'cross':
			print "\nPlotting cross-validation results for best settings..."
			self.getCrossValidationResults(dict(), setting_dict['tau10'], setting_dict['tau20'], 
											setting_dict['sigma_multiplier'], setting_dict['mu_multiplier'],
											save_plots=True)

		
		print "\nRetraining on training data with the best settings..."
		self.initializeHBLRModel(self.train_tasks)
		self.classifier.verbose = True
		self.setClassifierToSetting(setting_dict['tau10'], setting_dict['tau20'], setting_dict['sigma_multiplier'], setting_dict['mu_multiplier'])
		self.classifier.trainUntilConverged()
		
		print "\nPlotting and saving cool stuff about the final model..."
		self.saveImagePlot(self.classifier.phi, 'Phi')
		pd.DataFrame(self.classifier.phi).to_csv(self.results_path + self.save_prefix + "-phi.csv")
		self.saveConvergencePlots()

		print "\nEvaluating results on held-out test set!! ..."
		all_preds = []
		all_true_y = []
		all_X_data = []
		per_task_accs = [np.nan] * self.n_tasks
		per_task_aucs = [np.nan] * self.n_tasks
		per_task_f1 = [np.nan] * self.n_tasks
		per_task_precision = [np.nan] * self.n_tasks
		per_task_recall = [np.nan] * self.n_tasks
		for t in range(self.n_tasks):
			preds = self.classifier.predictBinary(self.test_tasks[t]['X'], t)
			true_y = list(self.test_tasks[t]['Y'].flatten())

			if len(preds)==0 or len(true_y) == 0:
				continue

			all_preds.extend(preds)
			all_true_y.extend(true_y)
			all_X_data.extend(self.test_tasks[t]['X'])

			# save the per-task results
			t_acc, t_auc, t_f1, t_precision, t_recall = helper.computeAllMetricsForPreds(preds, true_y)
			per_task_accs[t] = t_acc
			per_task_aucs[t] = t_auc
			per_task_f1[t] = t_f1
			per_task_precision[t] = t_precision
			per_task_recall[t] = t_recall

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
				if not self.users_as_tasks: task_name=helper.getFriendlyLabelName(task_name)
				print "\t\t", task_name, "- Acc:", per_task_accs[t], "AUC:", per_task_aucs[t], 'F1:', per_task_f1[t], 'Precision:', per_task_precision[t], 'Recall:', per_task_recall[t]

		if self.test_csv_filename is not None:
			print "\tSAVING HELD OUT PREDICITONS"
			if self.users_as_tasks:
				task_column = 'user_id'
				label_name = helper.getFriendlyLabelName(self.file_prefix)
				wanted_label = helper.getOfficialLabelName(label_name)
				predictions_df = helper.get_test_predictions_for_df_with_task_column(
						self.classifier.predictBinary, self.test_csv_filename, task_column, self.test_tasks, 
						wanted_label=wanted_label, num_feats_expected=np.shape(self.test_tasks[0]['X'])[1], 
						label_name=label_name, tasks_are_ints=False)
			else:
				predictions_df = helper.get_test_predictions_for_df_with_no_task_column(self.classifier.predictBinary, 
				self.test_csv_filename, self.test_tasks, num_feats_expected=np.shape(self.test_tasks[0]['X'])[1])
			predictions_df.to_csv(self.results_path + "Preds-" + self.save_prefix + '.csv')
		else:
			print "Uh oh, the test csv filename was not set, can't save test preds"

		print "\t SAVING CLASSIFIER"
		with open(self.results_path + "PickledModel-" + self.save_prefix + '.p',"w") as f:
			pickle.dump(self.classifier,f)

	def saveHintonPlot(self, matrix, num_tests, max_weight=None, ax=None):
		"""Draw Hinton diagram for visualizing a weight matrix."""
		fig,ax = plt.subplots(1,1)
		
		if not max_weight:
			max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

		ax.patch.set_facecolor('gray')
		ax.set_aspect('equal', 'box')
		ax.xaxis.set_major_locator(plt.NullLocator())
		ax.yaxis.set_major_locator(plt.NullLocator())

		for (x, y), w in np.ndenumerate(matrix):
			color = 'white' if w > 0 else 'black'
			size = np.sqrt(np.abs(0.5*w/num_tests)) # Need to scale so that it is between 0 and 0.5
			rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
								 facecolor=color, edgecolor=color)
			ax.add_patch(rect)

		ax.autoscale_view()
		ax.invert_yaxis()
		plt.savefig(self.figures_path + self.save_prefix + '-Hinton.eps')
		plt.close()

	def plotAccuracyAucAndClusters(self, accs, aucs, clusters):
		fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,10))
		ax1.hist(accs)
		ax1.set_title("Accuracy")
		ax2.hist(aucs)
		ax2.set_title("AUC")
		ax3.hist(clusters)
		ax3.set_title("Number of Clusters (K)")
		plt.savefig(self.figures_path + self.save_prefix + '-AccAucClusters.eps')
		plt.close()

	def saveConvergencePlots(self):
		hblr.plotConvergence(self.classifier.xi_convergence_list, 'Xi convergence', save_path=self.figures_path + self.save_prefix + '-ConvergenceXi.eps')
		hblr.plotConvergence(self.classifier.theta_convergence_list, 'Theta convergence', save_path=self.figures_path +self.save_prefix + '-ConvergenceTheta.eps')
		hblr.plotConvergence(self.classifier.phi_convergence_list, 'Phi convergence', save_path=self.figures_path +self.save_prefix + '-ConvergencePhi.eps')

	def saveImagePlot(self, matrix, name):
		plt.figure()
		plt.imshow(matrix)
		plt.savefig(self.figures_path + self.save_prefix + "-" + name + ".eps")
		plt.close()

	def updateSameTaskMatrix(self, same_task_matrix):
		most_likely_cluster = np.argmax(self.classifier.phi,axis=1)
		for row_task in range(self.n_tasks):
			for col_task in range(self.n_tasks):
				if most_likely_cluster[row_task]==most_likely_cluster[col_task]:
					same_task_matrix[row_task,col_task]+=1
		return same_task_matrix

	
if __name__ == "__main__":
	print "HBLR MODEL SELECTION"
	print "\tThis code will sweep a set of parameters to find the ideal settings for HBLR for a single dataset"

	if len(sys.argv) < 3:
		print "Error: usage is python HBLRWrapper.py <file prefix> <users as tasks> <continue>"
		print "\t<file prefix>: e.g. datasetTaskList-Discard-Future-Group_ - program will look in the following directory for this file", DEFAULT_DATASETS_PATH
		print "\t<users as tasks>: type 'users' for users as tasks, or 'wellbeing' for wellbeing measures as tasks"
		print "\t<continue>: optional. If 'True', the wrapper will pick up from where it left off by loading a previous validation results file"
		print "\t<csv file for testing>: optional. If you want to get the final test results, provide the name of a csv file to test on"
		sys.exit()
	filename= sys.argv[1] #get data file from command line argument
	print "\nLoading dataset", DEFAULT_DATASETS_PATH + filename
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

	wrapper = HBLRWrapper(filename, users_as_tasks=users_as_tasks, cont=cont, test_csv_filename=csv_test_file)
	
	print "\nThe following parameter settings will be tested:"
	print "\ttau10:  	\t", wrapper.tau10s
	print "\ttau20:   	\t", wrapper.tau20s
	print "\tsigma multipliers: \t", wrapper.sigma_multipliers
	print "\tmu multipliers:    \t", wrapper.mu_multipliers

	print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'
	print "\nThe validation and testing figures will be saved in:", wrapper.figures_path + wrapper.save_prefix

	wrapper.run()
