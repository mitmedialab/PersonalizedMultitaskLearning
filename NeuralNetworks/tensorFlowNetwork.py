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
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

PATH_TO_DATA = '/Your/path/here/'

sys.path.append(PATH_TO_REPO)
import helperFuncs as helper

DEFAULT_BATCH_SIZE = 50
ACCURACY_LOGGED_EVERY_N_STEPS = 100
ACCURACY_OUTPUT_EVERY_N_STEPS = 1000
MINIMUM_STEPS = 2000
DEFAULT_VAL_TYPE = 'cross'
PRINT_CROSS_VAL_FOLDS = False

RESULTS_METRICS = ['acc', 'auc', 'f1', 'precision', 'recall']

def reloadHelper():
	reload(helper)

def weight_variable(shape,name):
	initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])))
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def getSoftmaxLoss(logits, labels):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def getSigmoidLoss(logits, labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def getL2RegularizationPenalty(weights, beta):
	return beta * sum([tf.nn.l2_loss(w) for w in weights])

def changeLabelsToOneHotEncoding(y, trinary=False):
	if trinary:
		y = y+1
		num_labels = 3
	else:
		num_labels = 2
	return (np.arange(num_labels) == y[:,None]).astype(np.float32)

def getOneHotAccuracy(predictions, labels):
	if len(predictions) == 0:
		print "no predictions, returning"
		return np.nan
	return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
		  / predictions.shape[0])

def getTensorFlowOneHotAccuracy(predictions,labels):
	correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(labels,1))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def getAccuracyForLabel(predictions, labels, wanted_labels, target_label, binary=True):
	threshold_preds,true_y = getBinaryPredsTrueYForLabel(predictions,labels,wanted_labels,target_label)
	num_correct = np.sum(true_y == threshold_preds)
	return (float(num_correct) / float(len(true_y))) * 100.0

def getBinaryPredsTrueYForLabel(predictions,labels,wanted_labels,target_label):
	label_idx = wanted_labels.index(target_label)
	pred_y = predictions[:,label_idx]
	true_y = labels[:,label_idx]

	return thresholdBinaryPredictions(pred_y), true_y

def getAucMultilabel(preds,labels,wanted_labels,target_label):
	binary_preds,true_y = getBinaryPredsTrueYForLabel(preds,labels,wanted_labels,target_label)
	return getAuc(binary_preds, true_y)

def getAuc(preds, true_y):
	try:
		return roc_auc_score(true_y, preds)
	except:
		return np.nan

def getAllMetricsForPredsOneHot(preds, true_y):
	acc = getOneHotAccuracy(preds, true_y)
	preds_flat = flattenOneHotPredictions(preds)
	y_flat = flattenOneHotPredictions(true_y)
	auc = helper.computeAuc(preds_flat,y_flat)
	f1 = helper.computeF1(preds_flat, y_flat)
	precision = helper.computePrecision(preds_flat, y_flat)
	recall = helper.computeRecall(preds_flat, y_flat)
	return acc, auc, f1, precision, recall

def getAllMetricsForLabel(preds,labels,wanted_labels,target_label):
	binary_preds,true_y = getBinaryPredsTrueYForLabel(preds,labels,wanted_labels,target_label)
	return helper.computeAllMetricsForPreds(binary_preds, true_y)

def flattenOneHotPredictions(preds):
	return np.argmax(preds, 1)

def thresholdBinaryPredictions(preds):
	y_hat = [0.0 if x <= .50 else 1.0 for x in preds]
	y_hat = np.asarray(y_hat)
	return y_hat.astype(np.float32)

def thresholdTrinaryPredictions(preds):
	preds = [0.0 if x <= 0.33 else x for x in preds]
	preds = [0.5 if (x > 0 and x <= 0.66) else x for x in preds ]
	return [1.0 if x >= 0.66 else x for x in preds ]

def makeMetricListDict():
	metric_list_dict = dict()
	for metric in RESULTS_METRICS:
		metric_list_dict[metric] = []
	return metric_list_dict

def _print_if_saved_setting_differs(class_var, setting_name, npz_file):
	if setting_name not in npz_file.keys():
		print "ERROR! The setting", setting_name, "is not in the saved model file."
		print "Using default value:", class_var
		print ""
		return False
	
	not_equal = False
	if type(class_var) is list:
		if len(class_var) != len(npz_file[setting_name]):
			not_equal = True
		else:
			for i in range(len(class_var)):
				if class_var[i] != npz_file[setting_name][i]:
					not_equal = True
	elif class_var != npz_file[setting_name]:
		not_equal = True
		
	if not_equal:
		print "WARNING! Saved setting for", setting_name, "is different!"
		print "\tModel's current value for", setting_name, "is", class_var
		print "\tBut it was saved as", npz_file[setting_name]
		print "Overwriting setting", setting_name, "with new value:", npz_file[setting_name]
		print ""
	return True

class TensorFlowNetwork:
	def __init__(self, data_df, wanted_feats, wanted_labels, multilabel=False, optimize_labels=None, 
				verbose=True, num_cross_folds=5, val_type=DEFAULT_VAL_TYPE):
		if multilabel and optimize_labels is None:
			print "ERROR! need to specify which labels to optimize over if you are doing multilabel classification"

		print "Normalizing, filling and randomizing data dataframe..."
		self.data_df = helper.normalizeAndFillDataDf(data_df, wanted_feats, wanted_labels)
		self.data_df = self.data_df.reindex(np.random.permutation(self.data_df.index))

		self.wanted_feats = wanted_feats
		self.wanted_labels = wanted_labels
		self.multilabel = multilabel
		self.optimize_labels = optimize_labels
		self.verbose = verbose

		print "\nBuilding training/validation/testing matrices from data dataframe..."
		self.train_X, self.train_y = helper.getTensorFlowMatrixData(self.data_df, wanted_feats, wanted_labels, dataset='Train', single_output=(not multilabel))
		self.val_X, self.val_y = helper.getTensorFlowMatrixData(self.data_df, wanted_feats, wanted_labels, dataset='Val', single_output=(not multilabel))
		self.test_X, self.test_y = helper.getTensorFlowMatrixData(self.data_df, wanted_feats, wanted_labels, dataset='Test', single_output=(not multilabel))

		if not multilabel:
			self.train_y = changeLabelsToOneHotEncoding(self.train_y)
			self.val_y = changeLabelsToOneHotEncoding(self.val_y)
			self.test_y = changeLabelsToOneHotEncoding(self.test_y)

		print "\tTrain:", np.shape(self.train_X), np.shape(self.train_y)
		print "\tVal:", np.shape(self.val_X), np.shape(self.val_y)
		print "\tTest:", np.shape(self.test_X), np.shape(self.test_y)

		self.input_size = np.shape(self.train_X)[1]
		self.output_size = np.shape(self.train_y)[1]

		if self.multilabel:
			self.loss_func = getSigmoidLoss
		else:
			self.loss_func = getSoftmaxLoss
	
		self.initializeStoredTrainingMetrics()

		#the following code supports performing cross validation:
		self.val_type = val_type
		if val_type == 'cross':
			self.num_cross_folds = num_cross_folds
			self.crossVal_X, self.crossVal_y = helper.generateCrossValSet(self.train_X, self.train_y, self.val_X, self.val_y, self.num_cross_folds)			

		#parameters that can be tuned
		self.l2_beta = 5e-4
		self.initial_learning_rate=0.01
		self.dropout=True
		self.decay=True
		self.decay_steps=10000
		self.decay_rate=0.95
		self.batch_size = DEFAULT_BATCH_SIZE
		self.optimizer = tf.train.AdamOptimizer #can also be tf.train.AdagradOptimizer or tf.train.GradientDescentOptimizer 
		
		#network structure and running stuff
		self.list_hidden_sizes = [1024]
		self.connection_types = ['full','full']
		self.n_steps = 4001

		# Tensorflow graph computation
		self.graph = None
		self.session = None
		self.saver = None
	
	def initializeStoredTrainingMetrics(self):
		if self.multilabel:
			self.training_val_results = dict()
			self.final_test_results = dict()
			for metric in RESULTS_METRICS:
				self.training_val_results[metric] = dict()
				self.final_test_results[metric] = dict()
				for label in self.wanted_labels:
					self.training_val_results[metric][label] = []
					self.final_test_results[metric][label] = []
		else:
			self.training_val_results = makeMetricListDict()
			self.final_test_results = makeMetricListDict()

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

	def setUpNetworkStructure(self, hidden_sizes, connection_types):
		''' Pass in a network description using the following variables:
				list_hidden_sizes:  a list containing the number of hidden neurons in each hidden layer
				connection_types:   a list indicating how each layer is connected to the one after it. 
									types can be 'full' or 'conv'
			Note that the length of connection_types and should be one more than the length 
			of list_hidden_sizes'''
		self.list_hidden_sizes = hidden_sizes
		self.connection_types = connection_types

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

	def logAllValMetricsMultilabel(self,acc, auc, f1, precision, recall,label):
		self.training_val_results['acc'][label].append(acc)
		self.training_val_results['auc'][label].append(auc)
		self.training_val_results['f1'][label].append(f1)
		self.training_val_results['precision'][label].append(precision)
		self.training_val_results['recall'][label].append(recall)

	def getOverallResults(self):
		if self.multilabel:
			accs = [0] * len(self.optimize_labels)
			aucs = [0] * len(self.optimize_labels)
			f1s = [0] * len(self.optimize_labels)
			precisions = [0] * len(self.optimize_labels)
			recalls = [0] * len(self.optimize_labels)

			for i in range(len(self.optimize_labels)):
				accs[i] = self.training_val_results['acc'][self.optimize_labels[i]][-1]
				aucs[i] = self.training_val_results['auc'][self.optimize_labels[i]][-1]
				f1s[i] = self.training_val_results['f1'][self.optimize_labels[i]][-1]
				precisions[i] = self.training_val_results['precision'][self.optimize_labels[i]][-1]
				recalls[i] = self.training_val_results['recall'][self.optimize_labels[i]][-1]
			return np.nanmean(accs), np.nanmean(aucs), np.nanmean(f1s), np.nanmean(precisions), np.nanmean(recalls)
		else:
			acc = self.training_val_results['acc'][-1]
			auc = self.training_val_results['auc'][-1]
			f1 = self.training_val_results['f1'][-1]
			precision = self.training_val_results['precision'][-1]
			recall = self.training_val_results['recall'][-1]

		return acc, auc, f1, precision, recall

	def plotValResults(self, save_path=None, label=None):
		if label is not None:
			accs = self.training_val_results['acc'][label]
			aucs = self.training_val_results['auc'][label]
		else:
			accs = self.training_val_results['acc']
			aucs = self.training_val_results['auc']
		plt.figure()
		plt.plot([i * ACCURACY_LOGGED_EVERY_N_STEPS for i in range(len(accs))], accs)
		plt.plot([i * ACCURACY_LOGGED_EVERY_N_STEPS for i in range(len(aucs))], aucs)
		plt.xlabel('Training step')
		plt.ylabel('Validation accuracy')
		plt.legend(['Accuracy','AUC'])
		if save_path is None:
			plt.show()
		else:
			plt.savefig(save_path)
		plt.close()

	def initializeWeights(self, list_hidden_sizes):
		sizes = []
		self.weights = []
		self.biases = []
		for i in range(len(list_hidden_sizes)+1):
			if i==0:
				input_len = self.input_size
			else:
				input_len = list_hidden_sizes[i-1]
			
			if i==len(list_hidden_sizes):
				output_len = self.output_size
			else:
				output_len = list_hidden_sizes[i]
				
			layer_weights = weight_variable([input_len, output_len],name='weights' + str(i))
			layer_biases = bias_variable([output_len], name='biases' + str(i))
			
			self.weights.append(layer_weights)
			self.biases.append(layer_biases)
			sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
		
		if self.verbose:
			print("Okay, making a neural net with the following structure:")
			print(sizes)

	def setUpGraph(self, init_metrics=True):
		self.graph = tf.Graph()

		with self.graph.as_default():

			# Input data. For the training data, we use a placeholder that will be fed
			# at run time with a training minibatch.
			self.tf_train_dataset = tf.placeholder(tf.float32, name="train_x")
			self.tf_train_labels = tf.placeholder(tf.float32, name="train_y")
			self.tf_valid_dataset = tf.constant(self.val_X)
			self.tf_test_dataset = tf.constant(self.test_X)
			
			self.dropout_keep_prob = tf.placeholder("float")
			self.global_step = tf.Variable(0)  # count the number of steps taken.
			if self.decay:
				self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, 
																self.decay_steps, self.decay_rate)
			else:
				self.learning_rate = self.initial_learning_rate
		   
			# Variables.
			self.initializeWeights(self.list_hidden_sizes)
			
			def networkStructure(input_X):
				hidden = input_X
				for i in range(len(self.weights)):
					with tf.name_scope('layer' + str(i)) as scope:
						if self.connection_types[i] == 'full':
							hidden = tf.matmul(hidden, self.weights[i]) + self.biases[i]
						
						if i < len(self.weights)-1:
							hidden = tf.nn.relu(hidden)
							hidden = tf.nn.dropout(hidden, self.dropout_keep_prob)  
				return hidden

			# Training computation
			self.logits = networkStructure(self.tf_train_dataset)
			with tf.name_scope('loss') as scope:
				self.loss = self.loss_func(self.logits, self.tf_train_labels)
				self.loss += getL2RegularizationPenalty(self.weights + self.biases, self.l2_beta)

			# Optimizer.
			with tf.name_scope("optimizer") as scope:
				self.opt_step = self.optimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

			# Predictions for the training, validation, and test data.
			with tf.name_scope("outputs") as scope:
				if self.multilabel:
					self.train_prediction = tf.nn.sigmoid(networkStructure(self.tf_train_dataset))
					self.valid_prediction = tf.nn.sigmoid(networkStructure(self.tf_valid_dataset))
					self.test_prediction = tf.nn.sigmoid(networkStructure(self.tf_test_dataset))
				else:
					self.train_prediction = tf.nn.softmax(networkStructure(self.tf_train_dataset))
					self.valid_prediction = tf.nn.softmax(networkStructure(self.tf_valid_dataset))
					self.test_prediction = tf.nn.softmax(networkStructure(self.tf_test_dataset))

			self.init = tf.global_variables_initializer()
			
		self.session = tf.Session(graph=self.graph)
		self.session.run(self.init)
		with self.graph.as_default():
			self.saver = tf.train.Saver()

		if init_metrics:
			self.initializeStoredTrainingMetrics()

	def runGraph(self, num_steps=None, trial_name='test', print_test=False, return_test_preds=False):
		if num_steps is None: num_steps = self.n_steps

		with self.graph.as_default():
			if self.verbose: print("Initialized")

			for step in range(num_steps):
				# Pick an offset within the training data, which has been randomized.
				# Note: we could use better randomization across epochs.
				
				# Generate a minibatch.
				if len(self.train_y) <= self.batch_size:
					batch_data = self.train_X
					batch_labels = self.train_y
				else:
					try:
						offset = (step * self.batch_size) % (self.train_y.shape[0] - self.batch_size)
						batch_data = self.train_X[offset:(offset + self.batch_size), :]
						batch_labels = self.train_y[offset:(offset + self.batch_size), :]
					except:
						print "It thinks it can't compute the offset."
						print "trainY.shape[0]", self.train_y.shape[0],
						print "self.batch_size", self.batch_size
						batch_data = self.train_X
						batch_labels = self.train_y


				# Prepare a dictionary telling the session where to feed the minibatch.
				# The key of the dictionary is the placeholder node of the graph to be fed,
				# and the value is the numpy array to feed to it.
				if self.dropout:
					dropout_prob = 0.5
				else:
					dropout_prob = 1.0
				feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels,
							 self.dropout_keep_prob: dropout_prob}

				_, l, predictions = self.session.run(
					[self.opt_step, self.loss, self.train_prediction], feed_dict=feed_dict)
				if (step % ACCURACY_LOGGED_EVERY_N_STEPS  == 0):
					train_preds, val_preds = self.session.run([self.train_prediction, self.valid_prediction],
															  feed_dict={self.tf_train_dataset : batch_data, 
															  			 self.dropout_keep_prob: 1.0})
					
					if self.multilabel:
						train_accs = []
						train_aucs = []
						
						for label in self.wanted_labels:
							acc, auc, f1, precision, recall = getAllMetricsForLabel(val_preds,self.val_y,self.wanted_labels,label)
							self.logAllValMetricsMultilabel(acc, auc, f1, precision, recall, label)

							train_accs.append(getAccuracyForLabel(train_preds, batch_labels, self.wanted_labels, label))
							train_aucs.append(getAucMultilabel(train_preds, batch_labels,self.wanted_labels,label))
						train_acc = np.mean(train_accs)
						train_auc = np.mean(train_aucs)
					else:
						acc, auc, f1, precision, recall = getAllMetricsForPredsOneHot(val_preds, self.val_y)
						self.logAllValMetrics(acc, auc, f1, precision, recall)

						train_acc = getOneHotAccuracy(train_preds, batch_labels)
						train_auc = getAuc(train_preds, batch_labels)

					if self.verbose and (step % ACCURACY_OUTPUT_EVERY_N_STEPS  == 0):
						print("Minibatch loss at step %d: %f" % (step, l))
						print ("Minibatch training accuracy:", train_acc, "AUC:", train_auc)
						if self.multilabel:
							val_acc, val_auc, val_f1, val_precision, val_recall = self.getOverallResults()
							print("Validation accuracy:", val_acc)
							print("Validation AUC:", val_auc)
						else:
							print("Validation accuracy:", self.training_val_results['acc'][-1])
							print("Validation AUC:", self.training_val_results['auc'][-1])

			if print_test or return_test_preds:
				test_preds = self.session.run(self.test_prediction, feed_dict={self.dropout_keep_prob: 1.0})
			
			if print_test:
				if self.multilabel:
					print("RESULTS ON HELD-OUT TEST SET:")
					for label in self.wanted_labels:
						acc, auc, f1, precision, recall = getAllMetricsForLabel(test_preds,self.test_y,self.wanted_labels,label)
						self.logFinalTestResultsForLabel(acc, auc, f1, precision, recall, label)
						print(label, "... Acc:", acc, "AUC:", auc, "F1", f1, "Precision", precision, "recall", recall) 
				else:
					acc, auc, f1, precision, recall = getAllMetricsForPredsOneHot(test_preds, self.test_y)
					self.logFinalTestResults(acc, auc, f1, precision, recall)
					print("RESULTS ON HELD-OUT TEST SET... Acc:", acc, "AUC:", auc, "F1", f1, "Precision", precision, "recall", recall) 
			
			if return_test_preds:
				return test_preds

	def predict(self, X):
		feed_dict = {self.tf_train_dataset : X, self.dropout_keep_prob: 1.0}
		preds = self.session.run(self.train_prediction, feed_dict)
		preds = np.argmax(preds, axis=1)
		return preds

	def get_preds_for_df(self):
		X = self.data_df[self.wanted_feats].as_matrix()
		preds = self.predict(X)
		assert len(preds) == len(self.data_df)
		preds_df = copy.deepcopy(self.data_df)

		for i,wanted_label in enumerate(self.wanted_labels):
			label_name = helper.getFriendlyLabelName(wanted_label)
			preds_df['test_pred_'+label_name] = preds

			test_df = preds_df[preds_df['dataset']=='Test']
			test_df = test_df.dropna(subset=[wanted_label], how='any')
			all_preds = test_df['test_pred_'+label_name].tolist()
			all_true = test_df[wanted_label].tolist()
			print "FINAL METRICS ON TEST SET for label", label_name, ":", helper.computeAllMetricsForPreds(all_preds, all_true)

		print "Predictions have been computed and are stored in dataframe."
		return preds_df

	def trainAndValidate(self):
		self.setUpGraph()
		self.runGraph()
		return self.getOverallResults()

	def trainAndCrossValidate(self):
		num_folds = min(self.num_cross_folds, len(self.crossVal_X))
		accs = []
		aucs = []
		f1s = []
		precisions = []
		recalls = []
		for f in range(num_folds):
			val_X = self.crossVal_X[f]
			val_Y = self.crossVal_y[f]
			train_folds_X = [self.crossVal_X[x] for x in range(num_folds) if x != f]
			train_folds_Y = [self.crossVal_y[x] for x in range(num_folds) if x != f]
			train_X = train_folds_X[0]
			train_Y = train_folds_Y[0]
			for i in range(1,len(train_folds_X)):
				train_X = np.concatenate((train_X,train_folds_X[i]))
				train_Y = np.concatenate((train_Y,train_folds_Y[i]))

			self.train_X = train_X
			self.train_y = train_Y
			self.val_X = val_X
			self.val_y = val_Y
			acc, auc, f1, precision, recall = self.trainAndValidate()
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
			
			if _print_if_saved_setting_differs(self.dropout, 'dropout', npz_file):
				self.dropout = npz_file['dropout']
			if _print_if_saved_setting_differs(self.list_hidden_sizes, 'list_hidden_sizes', npz_file):
				self.list_hidden_sizes = npz_file['list_hidden_sizes']
				self.setUpNetworkStructure(self.list_hidden_sizes, 'full')
			if _print_if_saved_setting_differs(self.l2_beta, 'l2_beta', npz_file):
				self.l2_beta = npz_file['l2_beta']

		self.setUpGraph(init_metrics=False)
		self.saver.restore(self.session, checkpoint_file)


