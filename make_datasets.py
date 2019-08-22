""" This file contains functions for converting a .csv dataset into the 
	'task dict list' format used by the rest of the code. The .csv file must 
	have a particular format, with columns like 'user_id', and outcome columns
	containing '_Label'. For an example, see the file 'example_data.csv'. 

	How to partition tasks:
		'users-as-tasks': The .csv file will be partioned such that predicting 
			the outcome of each user is one task.
		'labels-as-tasks': The .csv file will be partitioned such that 
			predicting related outcomes is each task (e.g. predicting stress
			is one task and predicting happiness is another)
"""

import numpy as np
import pandas as pd
import sklearn as sk
import sys
import os
import pickle
import random
import time
import copy
import argparse
import helperFuncs as helper
from sklearn.cross_validation import StratifiedShuffleSplit

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', type=str, default='/Your/path/here/')
parser.add_argument('--task_type', type=str, default='users', 
					help="How to partition related tasks; can be 'users' so "
						 "that predicting the outcome for each user is its own "
						 "task, or 'labels', so that predicting related "
						 "outcomes (like stress, happiness, etc) are their "
						 "own tasks.")
parser.add_argument('--target_label', type=str, 
					default='tomorrow_Happiness_Evening_Label',
					help="Outcome label to predict for each user in "
						 "users-as-tasks")
parser.add_argument('--group_users_on', type=str, 
					default='user_id',
					help="Name of column that indicates user or cluster ID "
						 "for partitioning users into tasks.")

def getDatasetCoreName(datafile):
	return datafile[8:-4]

def getLabelTaskListFromDataset(datafile, subdivide_phys=True):
	df = pd.DataFrame.from_csv(datafile)
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

def getUserTaskListFromDataset(datafile, target_label, suppress_output=False, 
							   group_on='user_id', subdivide_phys=False):
	df = pd.DataFrame.from_csv(datafile)
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
			print("Processing task", user)
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

if __name__ == '__main__':
	kwargs = vars(parser.parse_args())

	if kwargs['task_type'] == 'labels':
		print("Creating a label task-dict-list dataset where tasks are "
			  "predicting related outcome labels.")
		getLabelTaskListFromDataset(kwargs['datafile'])
	else:
		print("Creating a user task-dict-list dataset where tasks are "
			  "predicting the outcome of each different person (user).")
		getUserTaskListFromDataset(kwargs['datafile'], 
								   target_label=kwargs['target_label'],
							       group_on=kwargs['group_users_on'])