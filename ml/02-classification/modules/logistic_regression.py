import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import math

DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_N_STEPS = 2000
DEFAULT_LMD = 1

# np.random.seed(123)

class LogisticRegression:
	"""
	:param csv_path: path of the csv file containing the data we want to fit
	:param config: dictionary that contain the configuration of the linear regressor (hyperparameters, convergence condition, feature selection)
	"""
	def __init__(self, csv_path, config=None):
		"""
		:param learning_rate: learning rate value
		:param n_steps: number of epochs around gradient descent
		:param n_features: number of features involved in regression
		:param lmd: regularization factor (lambda)
		"""

		if config is None:
			config = {
				'learning_rate': 1e-2,
				'n_steps': 2000,
				'features_select': 'Size',
				'class': 'Price',
				'lmd': 1
			}
		
		# Parameters
		self.csv_path = csv_path
		self.config = config
		self.learning_rate = config['learning_rate'] if config['learning_rate'] != '' else DEFAULT_LEARNING_RATE
		self.n_steps = int(config['n_steps']) if config['n_steps'] != '' else DEFAULT_N_STEPS
		self.features_select = config['features_select']
		self.y_label = config['class']
		self.lmd = config['lmd'] if 'lmd' in config else DEFAULT_LMD

		# Placeholders
		self.X, self.y = None, None
		self.X_train, self.y_train = None, None
		self.X_valid, self.y_valid = None, None
		self.X_test, self.y_test = None, None
		self.theta = None
		self.cost_history = None
		self.theta_history = None
		self.X_mean_trainingset, self.X_std_trainingset  = None, None
		self.y_mean_trainingset, self.y_std_trainingset = None, None

		# SEMI-AUTOMATIC BEHAVIOR. Automatically load data, perform preprocessing, etc
		self._load_and_preprocess()
		self.m_samples = self.X.shape[0]							# get m and n_features
		self._dataset_split()
		self._normalize()
		self._add_bias_column()
		self.n_features = self.X_train.shape[1]						# update n_features after adding bias column
		self.lmd_vector = np.full(self.n_features, self.lmd)		# Generate vector lmd containing all lmd scalar except 0-th element which is 0
		self.lmd_vector[0] = 0

	def _load_and_preprocess(self):
		'''
		This loads the data from the csv as a pandas dataframe, it select the features in which we're interested and the labels, 
		it applies shuffling to the dataset, and finally returns the data as numpy arrays
		OSS: X can be a matrix! The columns are the features, the rows the samples
		'''
		dataset = pd.read_csv(self.csv_path)
		dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)		# shuffle all the sample to avoid group bias

		print(dataset.describe(), '\n')
		print("Data Types:\n", dataset.dtypes)
		print()
		numeric_dataset = dataset.select_dtypes(include=[np.number])
		print(numeric_dataset.corr())
		print()
		# Extract the features we're interested in and transform everything in numpy arrays
		self.features_list = [feature.strip() for feature in self.features_select.split(',')]
		self.X = dataset[self.features_list].values
		# Get the initial number of features, it will updated later (we'll also add +1 to express the bias col which has still not been added)
		self.n_features = self.X.shape[1] + 1
		# Extract the class
		self.y = dataset[self.y_label].values
		# Let's return the processed X, y in case you need to use it outside of the class
		return self.X, self.y

	def _dataset_split(self):
		# Hold-out splitting, 80% training+validation set, 20% test set
		train_valid_index = round(len(self.X)*0.8)
		X_train_valid = self.X[:train_valid_index]
		y_train_valid = self.y[:train_valid_index]
		self.X_test = self.X[train_valid_index:]
		self.y_test = self.y[train_valid_index:]
		# split the training+valid set into training set and validation, with hold-out 70/30
		valid_index = round(train_valid_index*0.7)
		self.X_train = X_train_valid[:valid_index]
		self.y_train = y_train_valid[:valid_index]
		self.X_valid = X_train_valid[valid_index:]
		self.y_valid = y_train_valid[valid_index:]
		# Let's still return this tuple in case you need it outside of the class
		return (self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test)

	def _normalize(self):
		'''
		This function apply normalization to the feature of the datasets (train, valid, test) using z-score normalization
		Remember that z-score normalization returns datas that have zero mean and stddev=1
		'''
		# lambda function to apply z-score normalization to a generic set of features X
		_normalize_features = lambda X: (X - self.X_mean_trainingset) / self.X_std_trainingset
		# compute mean and std of X
		self.X_mean_trainingset = self.X_train.mean(axis=0)
		self.X_std_trainingset = self.X_train.std(axis=0)
		# apply z-score normalization
		self.X_train = _normalize_features(self.X_train)
		self.X_valid = _normalize_features(self.X_valid)
		self.X_test = _normalize_features(self.X_test)

	def _add_bias_column(self):
		# Add bias column (of ones) to X_train, X_valid, X_test. ACHTUNG: the bias column must be added AFTER normalization
		add_bias = lambda X: np.c_[np.ones(X.shape[0]), X]
		self.X_train = add_bias(self.X_train)
		self.X_valid = add_bias(self.X_valid)
		self.X_test = add_bias(self.X_test)

	def _sigmoid(self, z):
		'''
		compute the sigmoid of input value
		:param z: an array-like with shape (m,) as input elements
		:return: an array-like with shape (m,). Values are in sigmoid range 0-1
		'''
		return 1 / (1 + np.exp(-z))
	
	def _predict_prob(self, X, theta):
		"""
		Perform a complete prediction about X samples (that is, it computes h_theta(X), so it returns the probability that the element appartain to class True)
		:param X: test sample with shape (m, n_features)
		:return: prediction with respect to X sample. The shape of return array is (m, )
		"""
		return self._sigmoid(np.dot(X, theta))
	
	def predict(self, X, theta, threshold=0.5):
		"""
		Perform a complete prediction about the X samples (this returns the predicted class)
		:param X: test sample with shape (m, n_features)
		:param threshold: threshold value to disambiguate positive or negative sample
		:return: prediction wrt X sample. The shape of return array is (m,)
		"""
		# Xpred =  np.c_[np.ones(X.shape[0]), X]
		Xpred = X
		return self._predict_prob(Xpred, theta) >= threshold				# For example if the probability is 0.9, 0.9>=0.5(threshold) -> prediction=True

	# For logistic regression the cost formula is different; this is also called CROSS ENTROPY FUNCTION in logistic regression
	def cost(self, X, y, theta):
		m = len(y)
		h_theta = self._predict_prob(X, theta)
		# Compute the two terms that are part of cross entropy function. Those two are vectors, size mx1
		# notice we're using element-wise product now
		term_1 = y * np.log(h_theta)
		term_2 = (1 - y) * np.log(1 - h_theta)
		# Final function is given by the summation of those two vectors, and then applying np.mean we get the summation of all the elements of the vectors divided by m
		# This implements the summation in the cross entropy formula! Instead of iterating over multiple addends, we leverage the power of vectorial functions
		loss = -np.mean(term_1 + term_2)
		return loss
 
	def fit(self, X=None, y=None, update_internal=True):
		"""
		Apply gradient descent in full batch mode
		:param X: training samples with bias. If none is passed, self.X_train is used
		:param y: training target values. If none is passed, self.y_train is used
		:return: history of evolution of cost and theta during training steps
		"""
		if X is None or y is None:
			X = self.X_train
			y = self.y_train
		
		m = len(X)

		# Initialize theta to a random value (uniform distribution, range 0-1)
		# cost_history, theta_history are just for plots purposes (they will be J(theta) and theta at every iteration), not needed for learning 
		theta = np.random.rand(self.n_features)
		cost_history = np.zeros(self.n_steps)
		theta_history = np.zeros((self.n_steps, self.n_features)) 
						
		# OSS: FOR LOGISTIC REGRESSION THE GRADIENT HAS THE SAME EXPRESSION OF LINEAR REGRESSION, BUT THE FORMULA FOR THE PREDICTION h_theta AND THE COST IS DIFFERENT.
		for step in range(0, self.n_steps):
			# Predict the probability that element appartain to class 1 (True) for all the samples. This returns a vector of probabilities
			# The prediction in logistic regression has a different meaning: it gives the probability that the i-th sample belong to class 1 (True)
			y_predict = self.predict(X, theta)	
			# Compute the errors (prediction - dataset label value). This will be a vector, for all the samples
			# NOTICE HOW VECTOR OPERATIONS LET YOU DO EVERYTHING REAL QUICK! NO NEED TO WRITE LOOPS HERE TO CYCLE ALL THE SAMPLES!
			error = y_predict - y
			regularization = self.lmd_vector * theta
			gradient = (1/m) * ( np.dot(X.T, error) + regularization )
			# update theta
			theta = theta - self.learning_rate * gradient
			# Save theta history
			theta_history[step, :] = theta.T
			# Compute cost function (or better, loss function)
			cost_history[step] = self.cost(X, y, theta)

		# Update the internal parameter of the model only if update_internal==True, otherwise just return the cost_history and theta_history
		if update_internal:
			self.theta = theta
			self.cost_history = cost_history
			self.theta_history = theta_history

		return cost_history, theta_history

	def fit_minibatch_gd(self, X=None, y=None, batch_size=10, update_internal=True):
		'''
		Fit the training dataset by employing a minibatch gradient descent (compromise between batch gd and stochastic gd)
		If the size of the batch is batch_size, we can think of the minibatch gd as number_batches = m_samples/batch_size rounds of batch gd (while the scan of single samples
		inside a single batch can be seen as a pure stochastic gd, where at each sample we update the parameters and the next sample is chosen randomly from the one available
		in the current batch)
		:return: theta_history and cost_history
		'''
		if X is None or y is None:
			X = self.X_train
			y = self.y_train

		m = len(y)
		theta = np.random.rand(self.n_features)											# Initialize theta to a random value (uniform distribution, range 0-1)
		cost_history_train = np.zeros(self.n_steps)
		theta_history = np.zeros((self.n_steps, self.n_features))						# Initialize theta history (zero fill)
		
		# Running through epochs
		for step in range(0, self.n_steps):
			cost = 0
			# Iterate through the various batches. Here the index i is the starting point of the current batch; i+batch_size-1 is the end of the batch
			for i in range(0, self.m_samples, batch_size):
				# Select the portion to create a batch from X_train, by slicing from i to i+batch_size (ex from 0 to 9, then from 10 to 19, then from 20 to 29...)
				X_i = self.X_train[i : i+batch_size]
				y_i = self.y_train[i : i+batch_size]
				y_predict_i = self.predict(X_i)
				error = y_predict_i - y_i
				# Each one of those iteration through a single batch, is like a small batch gd; but the size is not m_samples, is batch_size, so in formula u have 1/batch_size
				regularization = self.lmd_vector * theta
				gradient = (1/batch_size) * ( np.dot(X_i.T, error) + regularization )
				theta = theta - self.learning_rate * gradient
				cost += self.cost(X_i, y_i, theta)
			# Here current epoch (step) has finished running, so put the results in the history lists
			theta_history[step, :] = theta.T
			cost_history_train[step] = cost
		
		if update_internal==True:
			self.theta = theta
			self.cost_history = cost_history_train
			self.theta_history = theta_history

		return cost_history_train, theta_history

	def fit_stochastic_gradientdescent(self):
		pass
	
	