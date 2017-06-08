from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import matplotlib.pyplot as plt


class LSSVM:
	def __init__(self, C, kernel_func, debug=False):
		''' Least-squares svm : svm with squared loss and binary classification yi in {-1,1}
			C: 
			kernel_func:	function that takes 2 arguments (X1, X2) and returns kernel matrix of size (len(X1),len(X2))
		'''
		
		self.C = C
		self.kernel_func = kernel_func

		self.data = None
		self.y = None
		self.b = None
		self.alphas = None

		self.debug = debug

	def fit(self,X,y):
		''' Linear program
		        AX = b
		|0      Y.T         | * |   b   | = |0| 
		|Y   Omega+(1/C)*I  |   | alpha |   |1|

		Note Omega[i,j] = y[i]*y[j]*K(x[i],x[j])
		'''
		self.data = X


		# Make sure y is the right dimension
		y = np.atleast_2d(y)
		if np.shape(y)[0]==1:
			y = y.T

		self.y = y

		N = len(X) 

		K = self.kernel_func(self.data,self.data)

		Omega = np.dot(self.y,self.y.T)*K
		bottom_right = Omega+(1.0/self.C)*np.eye(N)

		assert np.shape(bottom_right)==(N,N), "The bottom left matrix is the wrong size"
		
		if self.debug:
			print "K",K
			print "K nans", np.sum(np.isnan(K))
		


		first_row = np.hstack([np.zeros((1,1)),self.y.T])
		bottom_mat = np.hstack([self.y,bottom_right])
		A = np.vstack([first_row,bottom_mat])
	
		b_vec = np.vstack([0,np.ones((N,1))])

		try:
			params,residuals, rank,s = np.linalg.lstsq(A,b_vec)
		except:
			print "\n------WARNING!!!  These parameters didn't converge!------\n"
			return False
		
		self.b = params[0]
		self.alphas = params[1:]

		return True
		

	def predict(self,test_data):
		assert (self.b is not None) and (self.alphas is not None), "Model not trained yet"
		
		K = self.kernel_func(self.data,test_data)

		alphaY = self.alphas*self.y

		y_hat = np.sign(np.dot(alphaY.T,K)+self.b)

		return y_hat[0]
