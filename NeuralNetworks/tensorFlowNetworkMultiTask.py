"""Implements a multi-task learning (MTL) neural network.

Each task gets a unique set of final layers that are specific to predicting the
outcome for each task. These unique final layers are connected to a set of 
layers that are shared among all tasks. 

To train the network, a batch from one task is sampled and used to update the 
private task weights, and take a gradient step on the shared task weights."""
import matplotlib
matplotlib.use('Agg')	
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os 
import math
import random
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

PATH_TO_DATA = '/Your/path/here/'

import helperFuncs as helper
import tensorFlowNetwork as tfnet

def reloadFiles():
	reload(helper)
	reload(tfnet)

DEFAULT_BATCH_SIZE = 50
MINIMUM_STEPS = 2000
DEFAULT_VAL_TYPE = 'cross'
PRINT_CROSS_VAL_FOLDS = True
RESULTS_METRICS = ['acc', 'auc', 'f1', 'precision', 'recall']
PENALIZE_TASK_WEIGHTS = False

class TensorFlowNetworkMTL:
	def __init__(self, train_task_dict_list, val_task_dict_list=None, test_task_dict_list=None,
				 print_per_task=False, optimize_labels=None, verbose=True, 
				 num_cross_folds=5, val_type=DEFAULT_VAL_TYPE, accuracy_logged_every_n=1000, 
				 accuracy_output_every_n=1000):
		self.train_tasks = copy.deepcopy(train_task_dict_list)
		self.val_tasks = copy.deepcopy(val_task_dict_list)
		self.test_tasks = copy.deepcopy(test_task_dict_list)
		self.n_tasks = len(self.train_tasks)
		self.print_per_task = print_per_task

		self.task_training_order = self.generateNewTrainingOrder()

		self.verbose = verbose
		self.optimize_labels = optimize_labels
		self.loss_func = tfnet.getSoftmaxLoss
	
		for i in range(self.n_tasks):
			self.train_tasks[i]['Y'] = tfnet.changeLabelsToOneHotEncoding(self.train_tasks[i]['Y'])
			if self.val_tasks is not None:
				self.val_tasks[i]['Y'] = tfnet.changeLabelsToOneHotEncoding(self.val_tasks[i]['Y'])
			if self.test_tasks is not None:
				self.test_tasks[i]['Y'] = tfnet.changeLabelsToOneHotEncoding(self.test_tasks[i]['Y'])

		if self.print_per_task:
			self.optimize_labels = []
			for i in range(self.n_tasks):
				self.optimize_labels.append(self.train_tasks[i]['Name'])

		self.initializeStoredTrainingMetrics()

		#the following code supports performing cross validation:
		self.val_type = val_type
		if val_type == 'cross':
			assert (self.val_tasks is not None)
			self.num_cross_folds = num_cross_folds
			print "Generating cross validation sets for each ppt"
			for i in range(self.n_tasks):
				self.train_tasks[i]['crossVal_X'], self.train_tasks[i]['crossVal_y'] = helper.generateCrossValSet(self.train_tasks[i]['X'], self.train_tasks[i]['Y'], self.val_tasks[i]['X'], self.val_tasks[i]['Y'], self.num_cross_folds, verbose=False)			

		self.input_size = helper.calculateNumFeatsInTaskList(self.train_tasks)
		self.output_size = np.shape(self.train_tasks[0]['Y'])[1]
		print "OUTPUT SIZE IS CALCULATED TO BE:", self.output_size

		#parameters that can be tuned
		self.l2_beta = 5e-4
		self.initial_learning_rate=0.0001
		self.decay=True
		self.batch_size = 10
		self.decay_steps=1000
		self.decay_rate=0.95
		self.optimizer = tf.train.AdamOptimizer #can also be tf.train.AdagradOptimizer or tf.train.GradientDescentOptimizer 
		self.dropout=True

		#network structure and running stuff
		self.hidden_sizes_shared = [1024]
		self.hidden_size_task = 10
		self.connection_types_shared = ['full']
		self.connection_types_task = ['full','full']
		self.n_steps = 4001
		self.accuracy_logged_every_n = accuracy_logged_every_n
		self.accuracy_output_every_n = accuracy_output_every_n

		# Tensorflow graph computation
		self.graph = None
		self.session = None
		self.saver = None

		#Note: for now you can only have one layer of weights that is unique to the task (hidden_size_task is a scalar)
		#TODO: improve this later
		self.task_w1 = None
		self.task_b1 = None
		self.task_w2 = None
		self.task_b2 = None

	def generateNewTrainingOrder(self):
		return np.random.choice(len(self.train_tasks), len(self.train_tasks), replace=False)

	def setValTasks(self, val_tasks):
		self.val_tasks = val_tasks
		self.val_tasks[i]['Y'] = tfnet.changeLabelsToOneHotEncoding(self.val_tasks[i]['Y'])
					
	def setTestTasks(self, test_tasks):
		self.test_tasks = test_tasks
		self.test_tasks[i]['Y'] = tfnet.changeLabelsToOneHotEncoding(self.test_tasks[i]['Y'])

	def initializeStoredTrainingMetrics(self):
		self.val_nan_percent = []
		self.train_nan_percent = []
		self.training_val_results = tfnet.makeMetricListDict()
		self.final_test_results = tfnet.makeMetricListDict()
		if self.print_per_task:
			self.training_val_results_per_task = dict()
			self.final_test_results_per_task = dict()
			for metric in RESULTS_METRICS:
				self.training_val_results_per_task[metric] = dict()
				self.final_test_results_per_task[metric] = dict()
				for label in self.optimize_labels:
					self.training_val_results_per_task[metric][label] = []
					self.final_test_results_per_task[metric][label] = []

	def setLoss(self, loss_func):
		self.loss_func = loss_func

	def setParams(self, l2_beta=None, initial_learning_rate=None, dropout=None,
				decay=None, decay_steps=None, decay_rate=None, batch_size=None,
				early_termination=None, optimizer=None, n_steps=None):
		'''does not override existing parameter settings if the parameter is not set'''
		self.l2_beta = np.float32(l2_beta) if l2_beta is not None else self.l2_beta
		self.initial_learning_rate = np.float32(initial_learning_rate) if initial_learning_rate is not None else self.initial_learning_rate
		self.decay= decay if decay is not None else self.decay
		self.decay_steps= np.float32(decay_steps) if decay_steps is not None else self.decay_steps
		self.decay_rate= np.float32(decay_rate) if decay_rate is not None else self.decay_rate
		self.batch_size = batch_size if batch_size is not None else self.batch_size
		self.optimizer = optimizer if optimizer is not None else self.optimizer
		self.n_steps = n_steps if n_steps is not None else self.n_steps
		self.dropout = dropout if dropout is not None else self.dropout

	def setUpNetworkStructure(self, hidden_sizes_shared, hidden_size_task, connection_types_shared, 
							connection_types_task):
		''' Pass in a network description using the following variables:
				hidden_sizes_shared corresponds to the first layer of hidden nodes shared by all val_tasks
				hidden_sizes_task corresponds to the number of hidden nodes private to each task
				list_hidden_sizes_<mod>:  a list containing the number of hidden neurons in each hidden layer
				connection_types:   a list indicating how each layer is connected to the one after it. 
									types can be 'full' or 'conv'
			Note that the length of connection_types and should be one more than the length 
			of list_hidden_sizes'''
		self.hidden_sizes_shared = hidden_sizes_shared
		self.hidden_size_task = hidden_size_task
		self.connection_types_shared = connection_types_shared
		self.connection_types_task = connection_types_task

	def logValNans(self,nans):
		self.val_nan_percent.append(nans)

	def logTrainNans(self,nans):
		self.train_nan_percent.append(nans)

	def logFinalTestResults(self, acc, auc, f1, precision, recall):
		self.final_test_results['acc'].append(acc)
		self.final_test_results['auc'].append(auc)
		self.final_test_results['f1'].append(f1)
		self.final_test_results['precision'].append(precision)
		self.final_test_results['recall'].append(recall)

	def logFinalTestResultsForLabel(self, acc, auc, f1, precision, recall, label):
		self.final_test_results['acc'][label].append(acc)
		self.final_test_results['auc'][label].append(auc)
		self.final_test_results['f1'][label].append(f1)
		self.final_test_results['precision'][label].append(precision)
		self.final_test_results['recall'][label].append(recall)

	def logAllValMetrics(self, acc, auc, f1, precision, recall):
		self.training_val_results['acc'].append(acc)
		self.training_val_results['auc'].append(auc)
		self.training_val_results['f1'].append(f1)
		self.training_val_results['precision'].append(precision)
		self.training_val_results['recall'].append(recall)

	def logAllValMetricsPerTask(self,acc, auc, f1, precision, recall,label):
		self.training_val_results_per_task['acc'][label].append(acc)
		self.training_val_results_per_task['auc'][label].append(auc)
		self.training_val_results_per_task['f1'][label].append(f1)
		self.training_val_results_per_task['precision'][label].append(precision)
		self.training_val_results_per_task['recall'][label].append(recall)

	def getOverallResults(self, average_over_tasks=False):
		if average_over_tasks:
			accs = [0] * len(self.optimize_labels)
			aucs = [0] * len(self.optimize_labels)
			f1s = [0] * len(self.optimize_labels)
			precisions = [0] * len(self.optimize_labels)
			recalls = [0] * len(self.optimize_labels)

			for i in range(len(self.optimize_labels)):
				accs[i] = self.training_val_results_per_task['acc'][self.optimize_labels[i]][-1]
				aucs[i] = self.training_val_results_per_task['auc'][self.optimize_labels[i]][-1]
				f1s[i] = self.training_val_results_per_task['f1'][self.optimize_labels[i]][-1]
				precisions[i] = self.training_val_results_per_task['precision'][self.optimize_labels[i]][-1]
				recalls[i] = self.training_val_results_per_task['recall'][self.optimize_labels[i]][-1]
			return np.nanmean(accs), np.nanmean(aucs), np.nanmean(f1s), np.nanmean(precisions), np.nanmean(recalls)
		else:
			acc = self.training_val_results['acc'][-1]
			auc = self.training_val_results['auc'][-1]
			f1 = self.training_val_results['f1'][-1]
			precision = self.training_val_results['precision'][-1]
			recall = self.training_val_results['recall'][-1]

		return acc, auc, f1, precision, recall

	def getTaskResults(self, task):
		acc = self.training_val_results_per_task['acc'][self.optimize_labels[task]][-1]
		auc = self.training_val_results_per_task['auc'][self.optimize_labels[task]][-1]
		f1 = self.training_val_results_per_task['f1'][self.optimize_labels[task]][-1]
		precision = self.training_val_results_per_task['precision'][self.optimize_labels[task]][-1]
		recall = self.training_val_results_per_task['recall'][self.optimize_labels[task]][-1]
		return acc, auc, f1, precision, recall

	def plotValResults(self, save_path=None, label=None):
		if label:
			accs = self.training_val_results_per_task['acc'][label]
			aucs = self.training_val_results_per_task['auc'][label]
		else:
			accs = self.training_val_results['acc']
			aucs = self.training_val_results['auc']
		plt.figure()
		plt.plot([i * self.accuracy_logged_every_n for i in range(len(accs))], accs)
		plt.plot([i * self.accuracy_logged_every_n for i in range(len(aucs))], aucs)
		plt.xlabel('Training step')
		plt.ylabel('Validation accuracy')
		plt.legend(['Accuracy','AUC'])
		if save_path is None:
			plt.show()
		else:
			plt.savefig(save_path)

	def getL2RegularizationPenalty(self):
		shared_weight_loss = sum([tf.nn.l2_loss(w) for w in self.weights_shared])
		shared_bias_loss = sum([tf.nn.l2_loss(w) for w in self.biases_shared])
		if PENALIZE_TASK_WEIGHTS:
			task_weight_loss = tf.nn.l2_loss(self.task_w1) + tf.nn.l2_loss(self.task_w2)
			task_bias_loss = tf.nn.l2_loss(self.task_b1) + tf.nn.l2_loss(self.task_b2)
			return self.l2_beta * (shared_weight_loss + shared_bias_loss +  task_weight_loss + task_bias_loss)
		else:
			return self.l2_beta * (shared_weight_loss + shared_bias_loss)

	def initializeWeights(self):
		shared_sizes = []
		self.weights_shared = []
		self.biases_shared = []
		for i in range(len(self.hidden_sizes_shared)):
			if i==0:
				input_len = self.input_size
			else:
				input_len = self.hidden_sizes_shared[i-1]
			
			output_len = self.hidden_sizes_shared[i]
				
			layer_weights = tfnet.weight_variable([input_len, output_len],name='weights' + str(i))
			layer_biases = tfnet.bias_variable([output_len], name='biases' + str(i))
			
			self.weights_shared.append(layer_weights)
			self.biases_shared.append(layer_biases)
			shared_sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
		
		task_initial_w1 = tf.truncated_normal([self.n_tasks,self.hidden_sizes_shared[-1],self.hidden_size_task], stddev=1.0 / math.sqrt(float(self.hidden_sizes_shared[-1])))
		self.task_w1 = tf.Variable(task_initial_w1, name="task_weight1")
		task_initial_b1 = tf.constant(0.1, shape=[self.n_tasks,self.hidden_size_task])
		self.task_b1 = tf.Variable(task_initial_b1, name="task_bias1")
		
		task_initial_w2 = tf.truncated_normal([self.n_tasks,self.hidden_size_task,self.output_size], stddev=1.0 / math.sqrt(float(self.hidden_size_task)))
		self.task_w2 = tf.Variable(task_initial_w2, name="task_weight2")
		task_initial_b2 = tf.constant(0.1, shape=[self.n_tasks,self.output_size])
		self.task_b2 = tf.Variable(task_initial_b2, name="task_bias2")

		if self.verbose:
			print "Okay, making a neural net with the following structure:"
			print "\tShared:", shared_sizes
			print "\tTask:", tf.shape(self.task_w1), "x", tf.shape(self.task_w2)

	def setUpGraph(self, init_metrics=True):
		self.graph = tf.Graph()
		
		with self.graph.as_default():

			# Input data. For the training data, we use a placeholder that will be fed
			# at run time with a training minibatch.
			self.tf_train_dataset = tf.placeholder(tf.float32, name="train_x")
			self.tf_train_labels = tf.placeholder(tf.float32, name="train_y")
			self.tf_eval_dataset = tf.placeholder(tf.float32)
			self.tf_task = tf.placeholder(tf.int32, name='task')
			self.dropout_keep_prob = tf.placeholder("float", name='dropout_prob')

			self.global_step = tf.Variable(0, trainable=False, name='global_step')  # count the number of steps taken.
			if self.decay:
				self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, 
																self.global_step, self.decay_steps, 
																self.decay_rate)
			else:
				self.learning_rate = self.initial_learning_rate
		   
			# Variables.
			self.initializeWeights()
			
			def networkStructure(input_X, task):
				hidden = input_X
				for i in range(len(self.weights_shared)):
					with tf.name_scope('layer' + str(i)) as scope:
						if self.connection_types_shared[i] == 'full':
							hidden = tf.matmul(hidden, self.weights_shared[i]) + self.biases_shared[i]
						
						if i < len(self.weights_shared)-1:
							hidden = tf.nn.relu(hidden)
							hidden = tf.nn.dropout(hidden, self.dropout_keep_prob)  

				task_w1 = tf.gather(self.task_w1,task)
				task_w2 = tf.gather(self.task_w2,task)
				task_weights = [task_w1, task_w2]
				task_b1 = tf.gather(self.task_b1,task)
				task_b2 = tf.gather(self.task_b2,task)
				task_biases = [task_b1, task_b2]
				
				for i in range(len(task_weights)):
					if self.connection_types_task[i] == 'full':
						hidden = tf.matmul(hidden, task_weights[i]) + task_biases[i]
					
					if i < len(task_weights)-1:
						hidden = tf.nn.relu(hidden)
						hidden = tf.nn.dropout(hidden, self.dropout_keep_prob)  
				
				return hidden

			# Training computation
			self.logits = networkStructure(self.tf_train_dataset, self.tf_task)
			with tf.name_scope('loss') as scope:
				self.loss = self.loss_func(self.logits, self.tf_train_labels)
				self.loss += self.getL2RegularizationPenalty()

			# Optimizer.
			with tf.name_scope("optimizer") as scope:
				self.opt_step = self.optimizer(self.learning_rate).minimize(self.loss, 
																			 global_step=self.global_step)

			# Predictions for the training, validation, and test data.
			with tf.name_scope("outputs") as scope:
				self.train_prediction = tf.nn.softmax(networkStructure(self.tf_train_dataset, self.tf_task))
				self.eval_prediction = tf.nn.softmax(networkStructure(self.tf_eval_dataset, self.tf_task))

			self.init = tf.global_variables_initializer()

		self.session = tf.Session(graph=self.graph)
		self.session.run(self.init)
		with self.graph.as_default():
			self.saver = tf.train.Saver()

		if init_metrics:
			self.initializeStoredTrainingMetrics()

	def runGraph(self, num_steps=None, cross_val_fold=None, print_test=False):
		if num_steps is None: num_steps = self.n_steps
		
		with self.graph.as_default():
			if self.verbose: print("Initialized")

			for step in range(num_steps):
				# Pick an offset within the training data, which has been randomized.
				# Note: we could use better randomization across epochs.
				if step % len(self.task_training_order) == 0: #completed another run through all users
					self.task_training_order = self.generateNewTrainingOrder()
				task_offset = step % len(self.task_training_order)

				# Generate a minibatch.
				if cross_val_fold is not None:
					train_X, train_Y = self.getTrainAndValDataFromTaskForCrossValFold(task_offset, cross_val_fold, only_train=True)
				else:
					train_X = self.train_tasks[task_offset]['X']
					train_Y = self.train_tasks[task_offset]['Y']
				if train_X is None or train_Y is None:
					continue

				this_batch_size = min(len(train_X), self.batch_size)
				train_X, train_Y = helper.partitionRandomSubset(train_X, train_Y, this_batch_size, replace=True, return_remainder=False)
			
				# Prepare a dictionary telling the session where to feed the minibatch.
				# The key of the dictionary is the placeholder node of the graph to be fed,
				# and the value is the numpy array to feed to it.
				if self.dropout:
					dropout_prob = 0.5
				else:
					dropout_prob = 1.0
				feed_dict = {self.tf_train_dataset : train_X, self.tf_train_labels : train_Y, 
							self.tf_task: task_offset, self.dropout_keep_prob: dropout_prob}

				_, l, predictions = self.session.run(
					[self.opt_step, self.loss, self.train_prediction], feed_dict=feed_dict)
				
				if (step % self.accuracy_logged_every_n  == 0):
					self.log_validation_results(step, cross_val_fold,l)
			
			self.log_validation_results(step, cross_val_fold, l) # ensure that the best validation results are always logged

			if print_test:
				return self.get_test_results()

	def predict(self, X, task):
		feed_dict = {self.tf_train_dataset: X, self.tf_task: task, self.dropout_keep_prob: 1.0}
		preds = self.session.run(self.train_prediction, feed_dict)
		preds = np.argmax(preds, axis=1)
		return preds

	def log_validation_results(self, step, cross_val_fold, loss):
		#TODO: improve efficiency by pre-allocating instead of appending
		train_y_hat = []
		train_y_true = []
		val_y_hat = []
		val_y_true = []

		for t in range(self.n_tasks):
			if cross_val_fold is not None:
				train_X, train_Y, val_X, val_Y = self.getTrainAndValDataFromTaskForCrossValFold(t, cross_val_fold)
			else:
				train_X = self.train_tasks[t]['X']
				train_Y = self.train_tasks[t]['Y']
				val_X = self.val_tasks[t]['X']
				val_Y = self.val_tasks[t]['Y']
			if train_X is None or val_X is None:
				continue

			train_preds = self.session.run(self.train_prediction, feed_dict={self.tf_train_dataset : train_X, 
																				self.tf_task: t, 
																				self.dropout_keep_prob: 1.0})
			val_preds = self.session.run(self.eval_prediction, feed_dict={self.tf_eval_dataset : val_X, 
																		self.tf_task: t, 
																		self.dropout_keep_prob: 1.0})
			
			if self.print_per_task:
				acc, auc, f1, precision, recall = tfnet.getAllMetricsForPredsOneHot(val_preds, val_Y)
				self.logAllValMetricsPerTask(acc, auc, f1, precision, recall, self.val_tasks[t]['Name'])
			
			train_y_true.extend(train_Y)
			val_y_true.extend(val_Y)
			train_y_hat.extend(train_preds)
			val_y_hat.extend(val_preds)

		
		train_y_true = np.array(train_y_true)
		train_y_hat = np.array(train_y_hat)
		val_y_true = np.array(val_y_true)
		val_y_hat = np.array(val_y_hat)
		
		#get rid of nans. Terminate early if there are too many
		num_nans_train = np.sum(np.isnan(train_y_hat))
		total_train = np.shape(train_y_hat)[0] * np.shape(train_y_hat)[1]
		nan_percent_train = (num_nans_train / float(total_train)) * 100.0
		num_nans_val = np.sum(np.isnan(val_y_hat))
		total_val = np.shape(val_y_hat)[0] * np.shape(val_y_hat)[1]
		nan_percent_val = (num_nans_val / float(total_val)) * 100.0
		self.logValNans(nan_percent_val)
		self.logTrainNans(nan_percent_train)
		if self.verbose:
			print "Having to get rid of", num_nans_train, "nans in the training predictions:", nan_percent_train, "%"
			print "\tand", num_nans_val, "nans in the validation predictions:", nan_percent_val, "%"
			
		if nan_percent_train > 40 or nan_percent_val > 40:
			print "TOO MANY NANS! Terminating early"
			return -1 
		
		#if not self.print_per_task:
		train_y_hat[np.isnan(train_y_hat)] = 0.5
		val_y_hat[np.isnan(val_y_hat)] = 0.5
	
		train_acc = tfnet.getOneHotAccuracy(train_y_hat, train_y_true)
		train_auc = tfnet.getAuc(train_y_hat, train_y_true)

		acc, auc, f1, precision, recall = tfnet.getAllMetricsForPredsOneHot(val_y_hat, val_y_true)
		self.logAllValMetrics(acc, auc, f1, precision, recall)

		# This is really bad coding practice because this will never be output unless 
		# accuracy_output_every_n is a multiple of accuracy_logged_every_n
		if self.verbose and (step % self.accuracy_output_every_n  == 0):
			print("\nMinibatch loss at step %d: %f" % (step, loss))
			print ("Training accuracy:", train_acc, "AUC:", train_auc)
			print("Validation accuracy:", self.training_val_results['acc'][-1])
			print("Validation AUC:", self.training_val_results['auc'][-1])
			if self.print_per_task:
				print "Validation results!"
				for label in self.optimize_labels:
					print "\t", label
					for metric in RESULTS_METRICS:
						print "\t\t", metric, self.training_val_results_per_task[metric][label][-1]
				print ""

	def get_test_results(self):
		print("RESULTS ON HELD-OUT TEST SET...")
		test_y_hat = []
		test_y_true = []

		test_accs = []
		test_aucs = []
		test_f1s = []
		test_precisions = []
		test_recalls = []

		for t in range(self.n_tasks):
			X = self.test_tasks[t]['X']
			y = self.test_tasks[t]['Y']
			if X is None or len(X) == 0 or y is None or len(y) == 0:
				continue

			preds = self.session.run(self.eval_prediction, feed_dict={self.tf_eval_dataset : X, 
																	self.tf_task: t, 
																	self.dropout_keep_prob: 1.0})

			acc, auc, f1, precision, recall = tfnet.getAllMetricsForPredsOneHot(preds, y)

			if self.print_per_task:
				print self.test_tasks[t]['Name']
				print "\tacc:", acc
				print "\tauc:", auc
				print "\tf1:", f1
				print "\tprecision:", precision
				print "\trecall:", recall					
			
			test_accs.append(acc)
			test_aucs.append(auc)
			test_f1s.append(f1)
			test_precisions.append(precision)
			test_recalls.append(recall)

			test_y_true.extend(y)
			test_y_hat.extend(preds)
				
		test_y_hat = np.array(test_y_hat)
		test_y_true = np.array(test_y_true)
		
		#get rid of nans
		num_nans_test = np.sum(np.isnan(test_y_hat))
		total_test = np.shape(test_y_hat)[0] * np.shape(test_y_hat)[1]
		nan_percent_test = (num_nans_test / float(total_test)) * 100.0
		print "Having to get rid of", num_nans_test, "nans in the testing predictions:", nan_percent_test, "%"
		test_y_hat[np.isnan(test_y_hat)] = 0.5
		
		print "\nHELD OUT TEST METRICS COMPUTED BY APPENDING ALL PREDS"
		acc, auc, f1, precision, recall = tfnet.getAllMetricsForPredsOneHot(test_y_hat, test_y_true)
		print("... Acc:", acc, "AUC:", auc, "F1:", f1, "Precision:", precision, "Recall:", recall) 

		print "\nHELD OUT TEST METRICS COMPUTED BY AVERAGING OVER TASKS"
		acc = np.nanmean(test_accs)
		auc = np.nanmean(test_aucs)
		f1 = np.nanmean(test_f1s)
		precision = np.nanmean(test_precisions)
		recall = np.nanmean(test_recalls)
		print("... Acc:", acc, "AUC:", auc, "F1:", f1, "Precision:", precision, "Recall:", recall) 

	def getTrainAndValDataFromTaskForCrossValFold(self, task, fold, only_train=False):
		crossVal_X = self.train_tasks[task]['crossVal_X']
		crossVal_y = self.train_tasks[task]['crossVal_y']
		num_folds = len(crossVal_X)
		if fold >= num_folds:
			if only_train: 
				return None, None
			else:
				return None, None, None, None

		train_folds_X = [crossVal_X[x] for x in range(num_folds) if x != fold]
		train_folds_Y = [crossVal_y[x] for x in range(num_folds) if x != fold]
		
		train_X = train_folds_X[0]
		train_Y = train_folds_Y[0]
		for i in range(1,len(train_folds_X)):
			train_X = np.concatenate((train_X,train_folds_X[i]))
			train_Y = np.concatenate((train_Y,train_folds_Y[i]))

		if only_train:
			return train_X, train_Y
		else:
			val_X = crossVal_X[fold]
			val_Y = crossVal_y[fold]
			return train_X, train_Y, val_X, val_Y

	def trainAndValidate(self, cross_val_fold=None):
		self.setUpGraph()
		self.runGraph(cross_val_fold=cross_val_fold)
		return self.getOverallResults()

	def trainAndCrossValidate(self):
		accs = []
		aucs = []
		f1s = []
		precisions = []
		recalls = []
		for f in range(self.num_cross_folds):
			acc, auc, f1, precision, recall = self.trainAndValidate(f)
			accs.append(acc)
			aucs.append(auc)
			f1s.append(f1)
			precisions.append(precision)
			recalls.append(recall)
		if PRINT_CROSS_VAL_FOLDS: print "\t\tPer-fold cross-validation accuracy: ", accs
		return np.nanmean(accs), np.nanmean(aucs), np.nanmean(f1s), np.nanmean(precisions), np.nanmean(recalls)

	def save_model(self, file_name, directory):
		"""Saves a checkpoint of the model and a .npz file with stored rewards.

		Args:
		file_name: String name to use for the checkpoint and rewards files.
		Defaults to self.model_name if None is provided.
		"""
		if self.verbose: print "Saving model..."
		
		save_dir = directory + file_name
		os.mkdir(save_dir)
		directory = save_dir + '/'

		save_loc = os.path.join(directory, file_name + '.ckpt')
		training_epochs = len(self.training_val_results) * self.accuracy_logged_every_n
		self.saver.save(self.session, save_loc, global_step=training_epochs)
		
		
		npz_name = os.path.join(directory, file_name + '-' + str(training_epochs))
		
		if not self.print_per_task:
			np.savez(npz_name,
					training_val_results=self.training_val_results,
					l2_beta=self.l2_beta,
					dropout=self.dropout,
					hidden_sizes_shared=self.hidden_sizes_shared,
					hidden_size_task=self.hidden_size_task)
		else:
			np.savez(npz_name,
					training_val_results=self.training_val_results,
					training_val_results_per_task=self.training_val_results_per_task,
					l2_beta=self.l2_beta,
					dropout=self.dropout,
					hidden_sizes_shared=self.hidden_sizes_shared,
					hidden_size_task=self.hidden_size_task)

	def load_saved_model(self, directory, checkpoint_name=None,
		npz_file_name=None):
		"""Restores this model from a saved checkpoint.

		Args:
		directory: Path to directory where checkpoint is located. If 
		None, defaults to self.output_dir.
		checkpoint_name: The name of the checkpoint within the 
		directory.
		npz_file_name: The name of the .npz file where the stored
		rewards are saved. If None, will not attempt to load stored
		rewards.
		"""
		print "-----Loading saved model-----"
		if checkpoint_name is not None:
			checkpoint_file = os.path.join(directory, checkpoint_name)
		else:
			checkpoint_file = tf.train.latest_checkpoint(directory)
		print "Looking for checkpoin in directory", directory

		if checkpoint_file is None:
			print "Error! Cannot locate checkpoint in the directory"
			return
		else:
			print "Found checkpoint file:", checkpoint_file

		if npz_file_name is not None:
			npz_file_name = os.path.join(directory, npz_file_name)
			print "Attempting to load saved reward values from file", npz_file_name
			npz_file = np.load(npz_file_name)

			self.training_val_results = npz_file['training_val_results'].item()
			if self.print_per_task:
				self.training_val_results_per_task = npz_file['training_val_results_per_task'].item()
			
			if tfnet._print_if_saved_setting_differs(self.dropout, 'dropout', npz_file):
				self.dropout = npz_file['dropout']
			if tfnet._print_if_saved_setting_differs(self.hidden_sizes_shared, 'hidden_sizes_shared', npz_file):
				self.hidden_sizes_shared = npz_file['hidden_sizes_shared']
				self.setUpNetworkStructure(self.hidden_sizes_shared, self.hidden_size_task, 'full','full')
			if tfnet._print_if_saved_setting_differs(self.hidden_size_task, 'hidden_size_task', npz_file):
				self.hidden_size_task = npz_file['hidden_size_task']
				self.setUpNetworkStructure(self.hidden_sizes_shared, self.hidden_size_task, 'full','full')
			if tfnet._print_if_saved_setting_differs(self.l2_beta, 'l2_beta', npz_file):
				self.l2_beta = npz_file['l2_beta']

		self.setUpGraph(init_metrics=False)
		self.saver.restore(self.session, checkpoint_file)
