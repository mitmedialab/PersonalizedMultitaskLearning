import numpy as np
import pandas as pd
import copy
import os
import pickle
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import ast
import tensorflow as tf 

NAN_FILL_VALUE = 0

def computeAuc(preds, true_y):
	try:
		return roc_auc_score(true_y, preds)
	except:
		return np.nan

def computeF1(preds, true_y):
	try: 
		if (1 not in true_y) or (1 not in preds):
			# F-score is ill-defined when there are no true samples
			# F-score is ill-defined when there are no predicted samples.
			return np.nan
		return f1_score(true_y, preds)
	except:
		return np.nan

#The precision is the ratio tp / (tp + fp) where tp is the number of 
#true positives and fp the number of false positives.
def computePrecision(preds, true_y):
	try:
		if (1 not in preds):
			#Precision is ill-defined when there are no predicted samples.
			return np.nan
		return precision_score(true_y, preds)
	except:
		return np.nan

#The recall is the ratio tp / (tp + fn) where tp is the number of true 
#positives and fn the number of false negatives. The recall is intuitively 
#the ability of the classifier to find all the positive samples.
def computeRecall(preds, true_y):
	try:
		if 1 not in true_y:
			# Recall is ill-defined and being set to 0.0 due to no true samples
			return np.nan
		return recall_score(true_y, preds)
	except:
		return np.nan

def computeDistanceFromBaseline(preds, true_y):
	if len(np.shape(preds)) > 1:
		print("ERROR! Baseline distance function not defined for multi-dimensional predictions")
		return np.nan
	baseline = getBaseline(true_y)
	acc = getBinaryAccuracy(preds,true_y)
	return acc - baseline

def computeAllMetricsForPreds(preds, true_y):
	acc = getBinaryAccuracy(preds,true_y)
	auc = computeAuc(preds, true_y)
	f1 = computeF1(preds, true_y)
	precision = computePrecision(preds, true_y)
	recall = computeRecall(preds, true_y)
	return acc, auc, f1, precision, recall

def checkTaskList(train_tasks):
	for t in range(len(train_tasks)):
		isValidTask(train_tasks,t)
	print("...done!")

def isValidTask(train_tasks, t, print_msgs=True):
	if train_tasks[t]['Y'] is None or train_tasks[t]['X'] is None:
		if print_msgs: print("Uh oh,", train_tasks[t]['Name'], "is None!!")
		return False
	elif len(train_tasks[t]['X']) == 0:
		if print_msgs: print("Uh oh,", train_tasks[t]['Name'], "has no data!")
		return False
	elif len(train_tasks[t]['X']) != len(train_tasks[t]['Y']):
		if print_msgs: print("Uh oh,", train_tasks[t]['Name'], 
							 "has messed up data! Lengths of X and Y don't match")
		return False
	return True

def getBootstrapSample(test_df):
	bootstrap_ix = np.random.choice(test_df.index,len(test_df))
		
	test_df = test_df.loc[bootstrap_ix]
	test_df = test_df.reset_index()
	test_df = test_df.drop('index',1)
	return test_df

def plotROC(auc_list,fpr_list,tpr_list):
	mean_tpr = 0.0
	mean_fpr = np.linspace(0,1,100)

	plt.figure(figsize=(5,5))

	for i in range(len(fpr_list)):
		mean_tpr += np.interp(mean_fpr, fpr_list[i], tpr_list[i])
		mean_tpr[0] = 0.0
		plt.plot(fpr_list[i], tpr_list[i], lw=1, label='ROC fold %d (area = %0.2f)' % (i, auc_list[i]))

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	mean_tpr /= len(fpr_list)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('')
	plt.legend(loc="lower right")
	plt.show()

	return mean_auc, mean_fpr, mean_tpr

def getBinaryAccuracy(pred,true_labels):
	assert len(pred)==len(true_labels)

	correct_labels = [1 for i in range(len(pred)) if pred[i]==true_labels[i]]
	try:
		return len(correct_labels)/float(len(pred))
	except:
		return np.nan

def getBaseline(Y):
	if type(Y) != list:
		Y = Y.tolist()
	percentTrue = float(Y.count(1.0)) / float(len(Y))
	if percentTrue < 0.5:
		return 1.0 - percentTrue
	else:
		return percentTrue

def getTaskListFileCoreName(file_prefix):
	dash_loc = file_prefix.find('-')
	return file_prefix[dash_loc:-1]

def loadPickledTaskList(datasets_path, file_prefix, dataset, reshape=False, fix_y=False):
	task_list = pickle.load(open(datasets_path + file_prefix + dataset + ".p","rb"))

	task_list = fixTaskListFile(task_list)

	if reshape:
		for i in range(len(task_list)):
			if task_list[i]["Y"] is not None:
				task_list[i]["Y"] = task_list[i]["Y"].reshape(-1,1)

	if fix_y:
		for t in range(len(task_list)):
			task_list[t]["Y"] = 2*task_list[t]["Y"]-1
	
	return task_list


def fixTaskListFile(task_list,debug=False):
	num_feats = calculateNumFeatsInTaskList(task_list)
	for i in range(len(task_list)):
		if task_list[i]["Y"] is None:
			if debug: print("Y for task", task_list[i]['Name'], 
							"is None, fixing")
			task_list[i]['Y'] = np.zeros((0))
		if task_list[i]['X'] is None:
			if debug: print("X for task", task_list[i]['Name'], 
							"is None, fixing")
			task_list[i]['X'] = np.zeros((0,num_feats))
	return task_list


def loadCrossValData(datasets_path, file_prefix, fold, reshape=True, fix_y=False):
	save_prefix = getTaskListFileCoreName(file_prefix)

	train_tasks = loadPickledTaskList(datasets_path, "CVFold" + str(fold) + save_prefix, "Train", reshape=reshape, fix_y=fix_y)
	val_tasks = loadPickledTaskList(datasets_path, "CVFold" + str(fold) + save_prefix, "Val", reshape=reshape, fix_y=fix_y)

	return train_tasks, val_tasks

def generateCrossValPickleFiles(datasets_path, file_prefix, num_cross_folds):
	save_prefix = getTaskListFileCoreName(file_prefix)

	if os.path.exists(datasets_path + "CVFold0" + save_prefix + "Train.p"):
		print("\nCross validation folds have already been created")
		return

	train_tasks = pickle.load(open(datasets_path + file_prefix + "Train.p","rb"))
	val_tasks =  pickle.load(open(datasets_path + file_prefix + "Val.p","rb"))
	
	print("\nGenerating cross validation sets")
	new_train_tasks = [0] * (num_cross_folds+1)
	new_val_tasks = [0] * num_cross_folds
	for f in range(num_cross_folds):
		new_train_tasks[f] = copy.deepcopy(train_tasks)
		new_val_tasks[f] = copy.deepcopy(val_tasks)
	new_train_tasks[num_cross_folds] = copy.deepcopy(train_tasks)

	n_tasks = len(train_tasks)
	for t in range(n_tasks):
		crossVal_X, crossVal_y = generateCrossValSet(train_tasks[t]['X'], train_tasks[t]['Y'], val_tasks[t]['X'], val_tasks[t]['Y'], num_cross_folds, verbose=False)			

		for f in range(num_cross_folds):
			train_X, train_Y, val_X, val_Y = getTrainAndValDataForCrossValFold(crossVal_X, crossVal_y, f)
			new_train_tasks[f][t]['X'] = train_X
			new_train_tasks[f][t]['Y'] = train_Y
			new_val_tasks[f][t]['X'] = val_X
			new_val_tasks[f][t]['Y'] = val_Y

		new_train_tasks[num_cross_folds][t]['X'],new_train_tasks[num_cross_folds][t]['Y'] = getFullTrain(crossVal_X, crossVal_y)

	for f in range(num_cross_folds):
		pickle.dump(new_train_tasks[f], open(datasets_path + "CVFold" + str(f) +  save_prefix + "Train.p","wb"))
		pickle.dump(new_val_tasks[f], open(datasets_path + "CVFold" + str(f) + save_prefix + "Val.p","wb"))
	pickle.dump(new_train_tasks[num_cross_folds], open(datasets_path + "CVFullTrain" + save_prefix + ".p","wb"))


def addKeepIndicesToCrossValPickleFiles(datasets_path, file_prefix, num_cross_folds, keep_percent):
	save_prefix = getTaskListFileCoreName(file_prefix)

	for f in range(num_cross_folds):
		task_dict_list = pickle.load(open(datasets_path + "CVFold" + str(f) + save_prefix + "Train.p","rb"))
		for t in range(len(task_dict_list)):
			if not 'KeepIndices' in task_dict_list[t] or task_dict_list[t]['KeepIndices'] is None:
				n = len(task_dict_list[t]['X'])
				keep_indices = np.random.choice(n, n*keep_percent, replace=False)
				task_dict_list[t]['KeepIndices'] = keep_indices
		pickle.dump(task_dict_list, open(datasets_path + "CVFold" + str(f) +  save_prefix + "Train.p","wb"))

def getTrainAndValDataForCrossValFold(crossVal_X, crossVal_y, fold, only_train=False):
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

	val_X = crossVal_X[fold]
	val_Y = crossVal_y[fold]
	return train_X, train_Y, val_X, val_Y

def containsEachLabelType(labels):
	'''	Checks if a set of labels contains all labels types (-1, 0, 1)'''
	return 1 in labels and 0 in labels

def containsEachSVMLabelType(labels):
	return -1 in labels and 1 in labels

def getFullTrain(crossVal_X, crossVal_y):
	full_X = crossVal_X[0]
	full_Y = crossVal_y[0]
	for i in range(1,len(crossVal_X)):
		full_X = np.concatenate((full_X,crossVal_X[i]))
		full_Y = np.concatenate((full_Y,crossVal_y[i]))
	return full_X, full_Y

def getFriendlyLabelName(col):
	if col is None:
		return ""
	if type(col) != str:
		return str(col)

	name = ""
	if 'Happiness' in col:
		name ='Happiness'
	elif 'Calmness' in col:
		name = 'Calmness'
	elif 'Health' in col:
		name = 'Health'
	if 'Morning' in col:
		name = 'Morning-' + name
	if 'tomorrow' in col:
		name = 'tomorrow-' + name
	elif 'yesterday' in col:
		name = 'yesterday-' + name

	return name	

def getOfficialLabelName(string):
	type_mod = 'Group'
	if 'Personal' in string:
		type_mod = 'Personal'

	if 'Happiness' in string:
		return 'tomorrow_'+type_mod+'_Happiness_Evening_Label'
	elif 'Calmness' in string:
		return 'tomorrow_'+type_mod+'_Calmness_Evening_Label'
	elif 'Health' in string:
		return 'tomorrow_'+type_mod+'_Health_Evening_Label'
	else:
		print("Error! Could not determine official label name")
		return None

def getMinutesFromMidnight(df, feature):
	time_deltas = pd.to_datetime(df[feature]) - pd.to_datetime(df['timestamp'])
	mins = [time / pd.Timedelta('1 minute') for time in time_deltas]
	return [time if not pd.isnull(time) else np.nan for time in mins]

def mergeDataframes(all_df, mod_df, mod_name, merge_type='inner',merge_keys=['user_id','timestamp']):
	print("Merging", mod_name)
	old_len = len(all_df)
	print("\tMerged df started with", old_len, "samples")
	print("\t", mod_name, "has", len(mod_df), "samples")
	all_df = pd.merge(all_df, mod_df, how=merge_type, on=merge_keys)
	print("\tMerged df now has", len(all_df), "samples")
	print(mod_name, "is missing at least", old_len - len(all_df), "samples")
	
	return all_df

def renameAllColsWithPrefix(df,prefix,remove_len=0):
	for feat in df.columns.values:
		if feat != 'user_id' and feat != 'timestamp':
			df = df.rename(columns={feat:prefix+feat[remove_len:]})
	return df

def normalizeColumns(df, wanted_feats):
	train_df = df[df['dataset']=='Train']
	for feat in wanted_feats:
		train_mean = np.mean(train_df[feat].dropna().tolist())
		train_std = np.std(train_df[feat].dropna().tolist())
		zscore = lambda x: (x - train_mean) / train_std
		df[feat] = df[feat].apply(zscore)
	return df

def findNullColumns(df, features):
	df_len = len(df)
	bad_feats = []
	for feat in features:
		null_len = len(df[df[feat].isnull()])
		if df_len == null_len:
			bad_feats.append(feat)
	return bad_feats

def removeNullCols(df, features):
	'''Must check if a column is completely null in any of the datasets. Then it will remove it'''
	train_df = df[df['dataset']=='Train']
	test_df = df[df['dataset']=='Test']
	val_df = df[df['dataset']=='Val']

	null_cols = findNullColumns(train_df,features)
	null_cols_test= findNullColumns(test_df,features)
	null_cols_val = findNullColumns(val_df,features)

	if len(null_cols) > 0 or len(null_cols_test) > 0 or len(null_cols_val) > 0:
		for feat in null_cols_test:
			if feat not in null_cols:
				null_cols.append(feat)
		for feat in null_cols_val:
			if feat not in null_cols:
				null_cols.append(feat)
		print("Found", len(null_cols), 
			  "columns that were completely null. Removing", null_cols)

		df = dropCols(df,null_cols)
		for col in null_cols:
			features.remove(col)
	return df, features

def generateWekaFile(X,Y,features,path,name):
	f = open(path + name + '.arff', 'w')
	f.write("@relation '" + name + "'\n\n")

	for feat in features:
		f.write("@attribute " + feat + " numeric\n")
	f.write("@attribute cluster {True,False}\n\n")

	f.write("@data\n\n")
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if np.isnan(X[i,j]):
				f.write("?,")
			else:
				f.write(str(X[i,j]) + ",")
		if Y[i] == 1.0 or Y[i] == True:
			f.write("True\n")
		else:
			f.write("False\n")

	f.close()

def getMatrixData(data_df, wanted_feats, wanted_labels, dataset=None,single_output=False):
	if dataset is not None:
		set_df = data_df[data_df['dataset']==dataset]
	else:
		set_df = data_df
	
	X = set_df[wanted_feats].astype(float).as_matrix()

	if single_output:
		y = set_df[wanted_labels[0]].tolist()
	else:
		y = set_df[wanted_labels].as_matrix()
	
	return X,y

def normalizeAndFillDataDf(df, wanted_feats, wanted_labels, suppress_output=False, remove_cols=True):
	data_df = normalizeColumns(copy.deepcopy(df), wanted_feats)
	if remove_cols:
		data_df, wanted_feats = removeNullCols(data_df, wanted_feats)

	if not suppress_output: print("Original data length was", len(data_df))
	data_df = data_df.dropna(subset=wanted_labels, how='any')
	if not suppress_output: print(
		"After dropping rows with nan in any label column, length is", 
		len(data_df))

	data_df = data_df.fillna(NAN_FILL_VALUE) #if dataset is already filled, won't do anything

	return data_df

def getSvmPartitionDf(data_df, wanted_feats, wanted_labels, dataset='Train'):
	set_df = data_df[data_df['dataset']==dataset]

	keep_cols = copy.deepcopy(wanted_feats)
	keep_cols.extend(wanted_labels)
	set_df = set_df[keep_cols]
	
	return set_df

def getTensorFlowMatrixData(data_df, wanted_feats, wanted_labels, dataset='Train',single_output=False):
	set_df = data_df[data_df['dataset']==dataset]
	
	X = set_df[wanted_feats].astype(float).as_matrix()

	if single_output:
		y = set_df[wanted_labels[0]].tolist()
	else:
		y = set_df[wanted_labels].as_matrix()
	
	X = convertMatrixToTensorFlowFriendlyFormat(X)
	y = convertMatrixToTensorFlowFriendlyFormat(y)

	return X,y

def convertMatrixToTensorFlowFriendlyFormat(X):
	X = np.asarray(X)
	X = X.astype(np.float32)
	return X

def dropCols(df,cols):
	for col in cols:
		df = df.drop(col, 1)
	return df

def convertTimestampViaString(row):
	return str(row['timestamp'])

def getMinutesFromMidnight(df, feature):
	time_deltas = pd.to_datetime(df[feature]) - pd.to_datetime(df['timestamp'])
	mins = [time / pd.Timedelta('1 minute') for time in time_deltas]
	return [time if not pd.isnull(time) else np.nan for time in mins]

def renameAllColsWithPrefix(df,prefix,remove_len=0):
	for feat in df.columns.values:
		if feat != 'user_id' and feat != 'timestamp':
			df = df.rename(columns={feat:prefix+feat[remove_len:]})
	return df

def combineFilesIntoDf(file_path, filenames, reset_index=False, drop_cols=None):
	df = None
	for filename in filenames:
		fdf = pd.DataFrame.from_csv(file_path + filename)
		
		if reset_index:
			fdf = fdf.reset_index()
				
		if df is None:
			df = fdf.copy(deep=True)
		else:
			df = pd.concat([df,fdf])
			
	if drop_cols is not None:
		for feat in drop_cols:
			df = df.drop(feat, 1)
	
	return df

def partitionRandomSubset(X, Y, size, replace=False, return_remainder=True):
	subset_indices = np.random.choice(len(X), size, replace=replace)
	
	sub_X = X[subset_indices]
	sub_Y = Y[subset_indices]

	if return_remainder:
		remainder_indices = [x for x in range(0,len(X)) if x not in subset_indices]
		remainder_X = X[remainder_indices]
		remainder_Y = Y[remainder_indices]
		return sub_X, sub_Y, remainder_X, remainder_Y
	else:
		return sub_X, sub_Y

def generateCrossValSet(train_X, train_y, val_X, val_y, num_cross_folds, verbose=True):
	if verbose:
		print("...generating cross validation folds...")

	fullTrain_X = np.concatenate((train_X,val_X))
	fullTrain_y = np.concatenate((train_y,val_y))
	if len(fullTrain_X) <= 1:
		print("LENGTH IS", len(fullTrain_X))
	crossVal_X = []
	crossVal_y = []

	size = int(len(fullTrain_X) / num_cross_folds)
	if size < 1:
		size = 1
	remainder_X = fullTrain_X
	remainder_y = fullTrain_y
	for i in range(num_cross_folds-1):
		sub_X, sub_y, remainder_X, remainder_y = partitionRandomSubset(remainder_X, remainder_y, size)
		crossVal_X.append(sub_X)
		crossVal_y.append(sub_y)
		if len(remainder_X) == 0:
			# Insufficient data to make all folds, returning remaining.
			return crossVal_X, crossVal_y
	crossVal_X.append(remainder_X)
	crossVal_y.append(remainder_y)

	return crossVal_X, crossVal_y

def discardNans(df,col1,col2):
    small_df = df[[col1,col2]]
    small_df = small_df.dropna()
    x = small_df[col1].tolist()
    y = small_df[col2].tolist()
    n = len(x)
    return x,y,n

def calcCorrelation(df,col1,col2):
    x,y,n = discardNans(df,col1,col2)
    return stats.pearsonr(x, y)

def calculateNumFeatsInTaskList(task_dict_list):
	i=0
	X = task_dict_list[i]['X']
	while len(X) == 0 and i < len(task_dict_list):
		i=i+1
		X = task_dict_list[i]['X']
	return np.shape(X)[1]

def addPredsToPredsDf(df, preds, true, task_name):
	assert len(preds) == len(true)

	for i in range(len(preds)):
		df = df.append({'task_name':task_name, 'prediction':preds[i], 
						'true':true[i]}, ignore_index=True)

	return df

def fixSettingDictLoadedFromResultsDf(setting_dict):
	if 'hidden_layers' in setting_dict.keys():
		if type(setting_dict['hidden_layers']) == str:
			setting_dict['hidden_layers'] = ast.literal_eval(setting_dict['hidden_layers'])

	if 'optimizer' in setting_dict.keys():
		if 'GradientDescent' in setting_dict['optimizer']:
			setting_dict['optimizer'] = tf.train.GradientDescentOptimizer
		elif 'Adagrad' in setting_dict['optimizer']:
			setting_dict['optimizer'] = tf.train.AdagradOptimizer
		else:
			setting_dict['optimizer'] = tf.train.AdamOptimizer

	for setting in ['batch_size','decay_steps']:
		if setting in setting_dict.keys():
			setting_dict[setting] = int(setting_dict[setting])

	return setting_dict

def get_secs_mins_hours_from_secs(total_secs):
	hours = total_secs / 60 / 60
	mins = (total_secs % 3600) / 60
	secs = (total_secs % 3600) % 60

	if hours < 1: hours = 0
	if mins < 1: mins = 0
	
	return hours, mins, secs

def tf_weight_variable(shape, name):
    """Initializes a tensorflow weight variable with random values 
    centered around 0.
    """
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float64)
    return tf.Variable(initial, name=name)

def tf_bias_variable(shape, name):
    """Initializes a tensorflow bias variable to a small constant value."""
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=name)

def get_test_predictions_for_df_with_task_column(model_predict_func, csv_path, task_column, tasks, 
												wanted_label=None, num_feats_expected=None, label_name="", 
												tasks_are_ints=True):
	data_df = pd.DataFrame.from_csv(csv_path)
	
	wanted_feats = [x for x in data_df.columns.values if x != 'user_id' and x != 'timestamp' and 'ppt_id' not in x and x!= 'dataset' and '_Label' not in x and 'Cluster' not in x]
	if num_feats_expected is not None and len(wanted_feats) != num_feats_expected:
		print("Error! Found", len(wanted_feats), 
			  "features but was expecting to find", num_feats_expected)
		return

	if wanted_label is not None:
		wanted_labels = [wanted_label]
	else:
		wanted_labels = [x for x in data_df.columns.values if '_Label' in x and 'tomorrow_' in x and 'Evening' in x and 'Alertness' not in x and 'Energy' not in x]

	data_df = normalizeAndFillDataDf(data_df, wanted_feats, wanted_labels)

	if label_name is "" and wanted_label is not None:
		label_name = getFriendlyLabelName(wanted_label)

	for i,task_dict in enumerate(tasks):
		task = task_dict['Name']
		if tasks_are_ints:
			task = int(task)
		task_df = data_df[data_df[task_column]==task]
		X = task_df[wanted_feats].as_matrix()
		preds = model_predict_func(X, i)
		data_df.loc[task_df.index.values,'test_pred_'+label_name] = preds

	print("Predictions have been computed and are stored in dataframe.")
	
	if wanted_label is not None and wanted_label in data_df.columns.values:
		test_df = data_df[data_df['dataset']=='Test']
		all_preds = test_df['test_pred_'+label_name].tolist()
		all_true = test_df[wanted_label].tolist()
		print("FINAL METRICS ON TEST SET:", 
			  computeAllMetricsForPreds(all_preds, all_true))
	else:
		print("Cannot print test results unless wanted_label is set correctly")

	return data_df

def get_test_predictions_for_df_with_no_task_column(model_predict_func, csv_path, tasks, 
													num_feats_expected=None):
	data_df = pd.DataFrame.from_csv(csv_path)
	
	wanted_feats = [x for x in data_df.columns.values if x != 'user_id' and x != 'timestamp' and x!= 'dataset' and '_Label' not in x and 'Cluster' not in x]
	if num_feats_expected is not None and len(wanted_feats) != num_feats_expected:
		print("Error! Found", len(wanted_feats), 
			  "features but was expecting to find", num_feats_expected)
		return

	for i,task_dict in enumerate(tasks):
		wanted_label = task_dict['Name']
		label_name = getFriendlyLabelName(wanted_label)
		label_df = normalizeAndFillDataDf(copy.deepcopy(data_df), wanted_feats, [wanted_label])
		
		X = label_df[wanted_feats].as_matrix()
		preds = model_predict_func(X, i)
		data_df.loc[label_df.index.values,'test_pred_'+label_name] = preds

		test_df = data_df[data_df['dataset']=='Test']
		test_df = test_df.dropna(subset=[wanted_label], how='any')
		all_preds = test_df['test_pred_'+label_name].tolist()
		all_true = test_df[wanted_label].tolist()
		print("FINAL METRICS ON TEST SET for label", label_name, ":", 
			  computeAllMetricsForPreds(all_preds, all_true))

	print("Predictions have been computed and are stored in dataframe.")
	
	return data_df
	