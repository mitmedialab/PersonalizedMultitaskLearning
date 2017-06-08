import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import sys
import os
import pickle

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)
import helperFuncs as helper

def reloadHelper():
	reload(helper)

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
class LR:
	def __init__(self, penalty='l2', C=0.01, tol=0.001, solver= 'liblinear'):
		#data features
		self.n_features = None
		self.train_X = []
		self.train_Y = []
		self.val_X = []
		self.val_Y = []
		self.test_X = []
		self.test_Y = []

		#classifier features
		self.penalty = penalty
		self.C = C
		self.tolerance = tol
		self.solver = solver
	
	def setTrainData(self, X, Y):
		self.train_X = X
		self.train_Y = Y

		self.n_features = self.train_X.shape[1]

	def setTestData(self, X, Y):
		self.test_X = X
		self.test_Y = Y

	def setPenalty(self, penalty):
		self.penalty = penalty

	def setC(self, C):
		self.C = C

	def setSolver(self, solver):
		self.solver = solver

	def setValData(self, X, Y):
		self.val_X = X
		self.val_Y = Y

	def train(self):
		self.classifier = LogisticRegression(penalty=self.penalty, C=self.C, tol=self.tolerance, solver=self.solver)
		self.classifier.fit(self.train_X, self.train_Y)

	def predict(self, X):
		return self.classifier.predict(X)

	def getScore(self, X, Y):
		#returns accuracy
		return self.classifier.score(X, Y)

	def getFPRandTPR(self,X,Y):
		probas_ = self.classifier.fit(self.train_X, self.train_Y).predict_proba(X)
		fpr, tpr, thresholds = roc_curve(Y, probas_[:, 1])
		return fpr, tpr

	def getAUC(self,X,Y):
		fpr, tpr = self.getFPRandTPR(X,Y)
		return auc(fpr,tpr)

	def saveClassifierToFile(self, filepath):
		s = pickle.dumps(self.classifier)
		f = open(filepath, 'w')
		f.write(s)

	def loadClassifierFromFile(self, filepath):
		f2 = open(filepath, 'r')
		s2 = f2.read()
		self.classifier = pickle.loads(s2)

