import numpy as np
import pandas as pd
import sklearn as sk
import sys
import os
import pickle
import random
import time
import copy
from sklearn.cross_validation import StratifiedShuffleSplit

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_RESULTS_PATH = '/Your/path/here/'
DEFAULT_DATASETS_PATH = '/Your/path/here/'
DEFAULT_FIGURES_PATH = '/Your/path/here/'

def getDatasetCoreName(datafile):
	return datafile[8:-4]

def getWellbeingTaskListFromDataset(datafile, data_path=PATH_TO_DATASETS, subdivide_phys=True):
	df = pd.DataFrame.from_csv(data_path + datafile)
	wanted_labels = [x for x in df.columns.values if '_Label' in x and 'tomorrow_' in x and 'Evening' in x and 'Alertness' not in x and 'Energy' not in x]
	wanted_feats = [x for x in df.columns.values if x != 'user_id' and x != 'timestamp' and x!= 'dataset' and x!='Cluster' and '_Label' not in x]

	core_name = getDatasetCoreName(datafile)

	modality_dict = getModalityDict(wanted_feats, subdivide_phys=subdivide_phys)
	
	for dataset in ['Train','Val','Test']:
		task_dict_list = []
		for target_label in wanted_labels: 
			mini_df = helper.normalizeAndFillDataDf(df, wanted_feats, [target_label], suppress_output=True)
			mini_df.reindex(np.random.permutation(mini_df.index))
				
			X,y = helper.getTensorFlowMatrixData(mini_df, wanted_feats, [target_label], dataset=dataset, single_output=True)
			task_dict = dict()
			task_dict['X'] = X
			task_dict['Y'] = y
			task_dict['Name'] = target_label
			task_dict['ModalityDict'] = modality_dict
			task_dict_list.append(task_dict)
		pickle.dump(task_dict_list, open(data_path + "datasetTaskList-" + core_name + "_" + dataset + ".p","wb"))

def getModalityDict(wanted_feats, subdivide_phys=False):
	modalities = list(set([getFeatPrefix(x, subdivide_phys=subdivide_phys) for x in wanted_feats]))
	mod_dict = dict()
	for modality in modalities:
		mod_dict[modality] = getStartIndex(wanted_feats, modality)
	return mod_dict

def getStartIndex(wanted_feats, modality):
	for i,s in enumerate(wanted_feats):
		if modality[0:4] == 'phys' and 'H' in modality and modality != 'physTemp':
			if modality + ':' in s:
				return i
		else:
			if modality + '_' in s:
				return i

def getFeatPrefix(feat_name, subdivide_phys=False):
	idx = feat_name.find('_')
	prefix = feat_name[0:idx]
	if not subdivide_phys or prefix != 'phys':
		return prefix
	else:
		idx = feat_name.find(':')
		return feat_name[0:idx]

def getUserTaskListFromDataset(datafile, target_label, datapath=PATH_TO_DATASETS, suppress_output=False, 
							   group_on='user_id', subdivide_phys=False):
	df = pd.DataFrame.from_csv(datapath + datafile)
	wanted_feats = [x for x in df.columns.values if x != 'user_id' and x != 'timestamp' and x!= 'dataset' and x!='classifier_friendly_ppt_id' and 'Cluster' not in x and '_Label' not in x]
	
	df = helper.normalizeAndFillDataDf(df, wanted_feats, [target_label], suppress_output=True)
	df = df.reindex(np.random.permutation(df.index))

	dataset_name = getDatasetCoreName(datafile)
	label_name = helper.getFriendlyLabelName(target_label)
	
	modality_dict = getModalityDict(wanted_feats, subdivide_phys=subdivide_phys)

	train_task_dict_list = []
	val_task_dict_list = []
	test_task_dict_list = []
	for user in df[group_on].unique(): 
		if not suppress_output:
			print "Processing task", user
		mini_df = df[df[group_on] == user]

		train_task_dict_list.append(constructTaskDict(user, mini_df, wanted_feats, target_label, modality_dict, 'Train'))
		val_task_dict_list.append(constructTaskDict(user, mini_df, wanted_feats, target_label, modality_dict, 'Val'))
		test_task_dict_list.append(constructTaskDict(user, mini_df, wanted_feats, target_label, modality_dict, 'Test'))

	if group_on == 'user_id':
		dataset_prefix = "datasetUserTaskList-"
	elif group_on == 'Cluster':
		dataset_prefix = 'datasetClusterTasks-'
	else:
		dataset_prefix = group_on
	pickle.dump(train_task_dict_list, open(datapath + dataset_prefix + dataset_name + "-" + label_name + "_Train.p","wb"))
	pickle.dump(val_task_dict_list, open(datapath + dataset_prefix + dataset_name + "-" + label_name + "_Val.p","wb"))
	pickle.dump(test_task_dict_list, open(datapath + dataset_prefix + dataset_name + "-" + label_name + "_Test.p","wb"))

	return dataset_prefix + dataset_name + "-" + label_name

def constructTaskDict(task_name, mini_df, wanted_feats, target_label, modality_dict, dataset):
	X,y = helper.getTensorFlowMatrixData(mini_df, wanted_feats, [target_label], dataset=dataset, single_output=True)
	task_dict = dict()
	task_dict['X'] = X
	task_dict['Y'] = y
	task_dict['Name'] = task_name
	task_dict['ModalityDict'] = modality_dict
	return task_dict