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
	Class to perform learning for a linear regression. It has all methods to be trained with different strategies
	and one method to produce a full prediction based on input samples (inference). 
	This is completed by the class Evaluation in the module evaluation.py that measure performances and various indicators.
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
		# Generate eventual polynomial features
		self._polynomial_features()
		# get m and n_features
		self.m_samples = self.X.shape[0]

		self._dataset_split()
		self._normalize()
		self._add_bias_column()
		self.n_features = self.X_train.shape[1]			# update n_features after adding bias column (and eventual polynomial features)

		# Generate vector lmd (that is, a vector of dimension n+1 containing all lmd, 
		# with the exception of 0-th element which must be zero, since regularization must not be applied to 0-th element)
		self.lmd_vector = np.full(self.n_features, self.lmd)
		self.lmd_vector[0] = 0

	def _load_and_preprocess(self):
		'''
		This loads the data from the csv as a pandas dataframe, it select the features in which we're interested and the labels, 
		it applies shuffling to the dataset, and finally returns the data as numpy arrays
		OSS: X can be a matrix! The columns are the features, the rows the samples
		'''
		# read dataset from csv filepath
		dataset = pd.read_csv(self.csv_path)
		# shuffling all the samples to avoid group bias (the index is fixed after the shuffle by using reset_index)
		dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

		# print dataset stats
		print(dataset.describe(), '\n')
		print("Data Types:\n", dataset.dtypes)
		print()
		# test the correlation on dataset. This generates a nxn matrix (where n is the number of features)
		# and it gives a number in the range [-1, 1] (correlation coefficient) that tells statistically 
		# how that feature is correlated to another. The elements on the diagonal of this matrix will be 1
		numeric_dataset = dataset.select_dtypes(include=[np.number])
		print(numeric_dataset.corr())
		print()

		# Extract from the dataset the features specified in self.features_select, put it into X and transform it into a numpy array
		# OSS: then with the .values attribute we transform the pandas dataframes in numpy arrays!
		# THIS IS ESSENTIAL IN ORDER TO OPERATE WITH THE DATA; YOU CAN'T OPERATE ON DATAFRAME AS NUMBERS
		self.features_list = [feature.strip() for feature in self.features_select.split(',')]
		self.X = dataset[self.features_list].values
		# Get the initial number of features, it will updated later (we'll also add +1 to express the bias col which has still not been added)
		self.n_features = self.X.shape[1] + 1
		# Extract from the dataset the label y and convert it into numpy array
		self.y = dataset[self.y_label].values
		# Let's return the processed X, y in case you need to use it outside of the class
		return self.X, self.y

	def _dataset_split(self):
		# in order to perform hold-out splitting 80/20 identify the index at 80% of the total length of the array
		# Before train_index we'll have the training set (80%); after we'll have the remaining 20% (test set)
		train_valid_index = round(len(self.X)*0.8)
		# split the training+valid set into training set and validation, with hold-out 70/30 and get index of the validation set (it will start after the 70% of the training set)
		valid_index = round(train_valid_index*0.7)

		# split the dataset into training+valid and test set. 
		X_train_valid = self.X[:train_valid_index]
		y_train_valid = self.y[:train_valid_index]
		# ACHTUNG: don't touch the test set. Don't base the z-score normalizzation also on data present on testset!
		self.X_test = self.X[train_valid_index:]
		self.y_test = self.y[train_valid_index:]

		# split training set into training and validation. 
		# Remember: the training set is the one on which you compute the weights/parameters;
		# the validation set is the one with which you compute the error of the hypothesis (regressor) you've generated;
		# in case this fails to obtain required performances, you can select a different model (the validation set is used for model selection)
		# Finally the test set is the one on which you test the selected model, and get the final performances results (rate of false positive, false negative, etc)
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
		# z-score normalization. Achtung: mean has another shape (it's a row), but by broadcasting (implicit operation)
		# In broadcasting the array "mean" here is replicated the same number of rows of X_train in order to let the arithmetic operation happen
		# Apply the same normalization to validation and test sets. THIS IS IMPORTANT! OTHERWISE PERFORMANCE METRICS WILL BE WRONG!
		# NOTICE HOW TO NORMALIZE THE FEATURES OF VALIDATION AND TEST SET WE USE THE MEAN AND STDDEV OF THE TRAINING SET
		self.X_mean_trainingset = self.X_train.mean(axis=0)
		self.X_std_trainingset = self.X_train.std(axis=0)
		self.y_mean_trainingset = self.y_train.mean(axis=0)
		self.y_std_trainingset = self.y_train.std(axis=0)
		_normalize_features = lambda X: (X - self.X_mean_trainingset) / self.X_std_trainingset
		################################################################################################################################################
		# ACHTUNG: The normalization must be applied to all the datasets (training, valid, test), in order to have data that span on the same scale
		# and are distributed within the same range.
		# However for the normalization, for all the datasets, you MUST use the mean and stddev of the training set, to avoid data leakage:
		# that means the model must not know any info from the validation or test set! 
		# So: NORMALIZATION IS APPLIED TO EACH DATASET (TRAIN, TEST, VALID) BUT THE MEAN AND STDDEV USED FOR THE NORMALIZATION ARE THE ONE COMPUTED 
		# FROM THE TRAINING SET
		################################################################################################################################################
		# Compute mean and std dev for each column on training set. 
		# ACHTUNG: to compute mean and stddev as column-wise opearation, you need to provide the parameter axis=0 (that means, perform the operation column-wise)
		# In numpy, axis=0 refers to the rows (so it means the mean and stddev operation scans the rows), while axis=1 refers to the columns
		# This operation makes sense since each column represent a feature, so by computing the mean of all the samples we'll have a row vector, where each element in 
		# the row represent the mean of the column of X having the same index
		# So mean, std will be both row vectors, where each element represent the mean/std of each corresponding column of X
		self.X_train = _normalize_features(self.X_train)
		self.X_valid = _normalize_features(self.X_valid)
		self.X_test = _normalize_features(self.X_test)
		# NORMALIZE TRAINING, VALIDATION AND TEST LABELS USING TRAINING SET LABELS' MEAN AND STD
		"""
		self.y_train = _normalize_labels(self.y_train)
		self.y_valid = _normalize_labels(self.y_valid)
		self.y_test = _normalize_labels(self.y_test)
		"""

	def _add_bias_column(self):
		# Add bias column (of ones) to X_train, X_valid, X_test. Remember this is needed when computing h_theta(x), so theta0 is included in the summation
		# This ensure we can compute h_theta(x) on all the datasets by simply using a matrix product
		# ACHTUNG: the bias column must be added AFTER normalization
		# The c_ attribute of a nparray is to create a nparray as concatenation of columns,
		# so we generate a nparray of ones of the same dimension of X_train, and then we concatenate this to the original X_train
		add_bias = lambda X: np.c_[np.ones(X.shape[0]), X]

		# Add it also to non-normalized version, needed in order to give directly the features (ex X_valid or X_test) as input to the predictor function
		self.X_train = add_bias(self.X_train)
		self.X_valid = add_bias(self.X_valid)
		self.X_test = add_bias(self.X_test)

	def _sigmoid(self, z):
		'''
		compute the sigmoid of input value
		:param z: an array-like with shape (m,) as input elements
		:return: an array-like with shape (m,). Values are in sigmoid range 0-1
		'''
		return 1 / (1 + math.exp(-z))
	
	def _predict_prob(self, X):
		"""
		Perform a complete prediction about X samples (that is, it computes h_theta(X))
		OSS: The prediction expression doesn't change if we compute thetas with or without regularization
		:param X: test sample with shape (m, n_features)
		:return: prediction with respect to X sample. The shape of return array is (m, )
		"""
		return self._sigmoid(np.dot(X, self.theta))
	
	def predict(self, X, threshold=0.5):
		"""
		Perform a complete prediction about the X samples
		:param X: test sample with shape (m, n_features)
		:param threshold: threshold value to disambiguate positive or negative sample
		:return: prediction wrt X sample. The shape of return array is (m,)
		"""
		# Xpred =  np.c_[np.ones(X.shape[0]), X]
		Xpred = X
		return self._predict_prob(Xpred) >= threshold				# For example if the probability is 0.9, 0.9>=0.5(threshold) -> prediction=True

	# For logistic regression the cost formula is different; this is also called CROSS ENTROPY FUNCTION in logistic regression
	def cost(self, X, y, theta):
		m = len(y)
		h_theta = self._predict_prob(X)
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
		FOR LOGISTIC REGRESSION THE GRADIENT HAS THE SAME FORMULA OF LINEAR REGRESSION, BUT 
		Apply gradient descent in full batch mode, without regularization, to training samples and return evolution
		history of train and validation cost
		OSS: This method updates the internal parameter of the model only if it's called without X,y parameters (so that it acts
		on the internal features/labels of the model, acquired from the csv file)
		If it's called with different X or y it will just return the cost_history and theta_history, without updating the internal model.
		This is usefull when we're performing a fit for debug purposes (for example when computing learning curves)
		:param X: training samples with bias. If none is passed, self.X_train is used
		:param y: training target values. If none is passed, self.y_train is used
		:return: history of evolution of cost and theta during training steps
		"""
		if X is None or y is None:
			X = self.X_train
			y = self.y_train

		# Get the number of samples of the dataset (that is, the number of rows of the matrix X). 
		# The first dimension of the matrix is always row, so len(X) returns number of rows!
		# OSS: If you want to get the number of columns (that is, the number of feature), you can use len(X[0])
		m = len(X)      

		# Initialize theta to a random value (uniform distribution, range 0-1)
		# cost_history, theta_history are just for plots purposes (they will be J(theta) and theta at every iteration), not needed for learning 
		theta = np.random.rand(self.n_features)
		cost_history = np.zeros(self.n_steps)
		theta_history = np.zeros((self.n_steps, self.n_features)) 
						
		# Here you should generate random parameters (a random vector theta of n_features elements) in order to init the theta vector,
		# but this has already been done in the __init__ method

		for step in range(0, self.n_steps):
			# Compute the hypothesis h_theta(x), for the current values of theta (random in the first run) and for all the samples in the training dataset
			# With a simple matrix product we'll get the predictons for all the samples (so for m sets of n features)
			# OSS: EVERY ITERATION OF THIS LOOP WILL BE AN EPOCH. SO THIS TRAINING IS CONSTITUTED OF self.n_steps EPOCHS.
			# NOTICE HOW BATCH GD USES THE ENTIRE DATASET, AT THE SAME TIME, TO COMPUTE THE PARAMETERS THETA, AND THE PARAMETERS ARE UPDATED ALL AT ONCE
			h_theta = self.predict(X)	# The prediction in logistic regression has a different meaning: it gives the probability that the i-th sample belong to class 1 
			# Compute the errors (prediction - dataset label value). This will be a vector, for all the samples
			# NOTICE HOW VECTOR OPERATIONS LET YOU DO EVERYTHING REAL QUICK! NO NEED TO WRITE LOOPS HERE TO CYCLE ALL THE SAMPLES!
			error = h_theta - y
			# Compute the gradient, with regularization (to do it without regularization it's sufficient to put lmd=0)
			# ANOTHER APPROACH HERE WOULD BE FOR THE REGULARIZATION TERM BE EQUAL TO (1/m)*(THETA.T * LMD_VEC), WHERE LMD_VEC IS A VECTOR CONTAINING 
			# ALL LAMBDA, EXCEPT THE 0-TH ELEMENT WHICH IS SET TO ZERO
			# ACHTUNG: THE PREDICTION IS NOT MODIFIED BY THE REGULARIZATION! THIS IS ONLY TO GIVE MORE STABILITY TO THE GRADIENT DESCENT
			regularization = self.lmd_vector * theta
			gradient = (1/m) * ( np.dot(X.T, error) + regularization )
			# UPDATE THETA. HERE WE CAN SEE HOW ALL THE COMPONENTS OF THETA (PARAMETERS) ARE UPDATED ALL AT ONCE. SO THE "DIRECTION" OF MOVEMENT IN THE THETA SPACE
			# COULD BE ANY DIRECTION! INSTEAD IN STOCHASTIC GD IN EACH UPDATE WE MOVE IN AN ALTERNATIVE MANNER AT FIRST ON THETA0 AXIS, THEN ON THETA1 AXIS, THEN THETA2 AXIS, ETC
			# MOREOVER WE ARE COMPUTING THE UPDATE USING THE ENTIRE DATASET; INSTEAD IN STOCHASTIC GD EACH UPDATE IS PERFORMED BY CONSIDERING A SINGLE SAMPLE
			theta = theta - self.learning_rate * gradient
			# Save theta history
			theta_history[step, :] = theta.T
			# Here we compute the cost function by computing the mean of the squared errors as a dot product 
			# (the error vector, dot the error vector trasposed).
			# The rest of the expression is for regularization (the expression of cost function with regularization applied is different from standard one)
			# ALTERNATIVE: Select all the components of theta besides the 0-th using slicing, and use the scalar version of lambda, self.lmd instead of the vector:
			# cost_history[step] = 1/(2*m) * ( np.dot(error.T, error) +  self.lmd * np.dot(theta.T[1:], theta[1:]) )
			cost_history[step] = self.cost(X, y, theta)

		# Update the internal parameter of the model only if update_internal==True, otherwise just return the cost_history and theta_history
		if update_internal:
			self.theta = theta
			self.cost_history = cost_history
			self.theta_history = theta_history

		return cost_history, theta_history

	def fit_minibatch_gd(self, X=None, y=None, batch_size=10, update_internal=True):
		'''
		Fit the training dataset (that is, generate the parameters theta) by employing a minibatch gradient descent (compromise between batch gd and stochastic gd)
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
				h_theta = np.dot(X_i, theta)
				error = h_theta - y_i
				# Each one of those iteration through a single batch, is like a small batch gd; but the size is not m_samples, is batch_size, so in formula u have 1/batch_size
				gradient = (1/batch_size) * ( np.dot(X_i.T, error) + (self.lmd_vector * theta) )
				theta = theta - self.learning_rate * gradient
				cost += 1/(2*batch_size) * np.dot(error.T, error)
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
	
	def plot_cost_training_history(self):
		plt.figure(figsize=(10, 6))
		# Plot cost history
		plt.plot(self.cost_history, 'g--')
		plt.xlabel('Iterations [i]')
		plt.ylabel('J(theta), cost function')
		plt.title('Cost History')
		plt.show()

	def plot_3d_cost(self):
		'''
		This generates the 3D plot of the gradient descent algorithm. So it plots the cost function J(theta0, theta1) which lives in 3D space, 
		and the curve (path) generated by the gradient descent algorithm (that is, the theta history)
		This only works for regression over single feature (for n_features>1 the number of parameters is greater than 2 and the plot lives in >3D space)
		'''
		if self.n_features != 2:
			print("3D cost plot only available for single feature (X must be a column vector, not a matrix)")
			return
		fig = plt.figure(figsize=(12, 8))
		ax = fig.add_subplot(111, projection='3d')
		
		# Generate a vector whose elements span from theta0.min to theta0.max; same for theta1
		theta0_range = np.linspace(self.theta_history[:, 0].min(), self.theta_history[:, 0].max(), 100)
		theta1_range = np.linspace(self.theta_history[:, 1].min(), self.theta_history[:, 1].max(), 100)
		# With the datapoints above for theta0, theta1, generate a mesh grid. This is the standard technique in numpy to evaluate scalar or vector fields.
		theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)
		
		# Zero-fill the matrix where we'll put the J(theta0, theta1) values
		J_vals = np.zeros(theta0_mesh.shape)
		# i, j here are integers. theta0_mesh.shape[0] is the length of the 
		for i in range(theta0_mesh.shape[0]):
			for j in range(theta0_mesh.shape[1]):
				# build the theta=(theta0, theta1) vector across the meshgrid
				theta = np.array([theta0_mesh[i, j], theta1_mesh[i, j]])
				# Compute the values of J(theta) in the given theta point, that will correspond to (i,j)
				# So there's a mapping (i, j) -> (theta0, theta1)
				J_vals[i, j] = self.cost(self.X_train, self.y_train, theta)
		
		# Plot the surface of J(theta)
		surf = ax.plot_surface(theta0_mesh, theta1_mesh, J_vals, cmap=cm.coolwarm, alpha=0.6)
		# Plot the path in the (theta0, theta1, J(theta)) space taken by the gradient descent algorithm
		ax.plot(self.theta_history[:, 0], self.theta_history[:, 1], self.cost_history, 'r-', label='Gradient descent path')
		
		ax.set_xlabel('Theta0')
		ax.set_ylabel('Theta1')
		ax.set_zlabel('Cost J(theta)')
		ax.set_title('3D visualization of cost function and gradient descent path')
		plt.colorbar(surf)
		plt.legend()
		plt.show()

	def gd_contour_plot(self):
		'''
		Same as plot_3d_cost, but it generate a contour plot instead of a 3D visual
		'''
		# Only works for single feature
		if self.n_features != 2:
			print("Contour plot only available for single feature")
			return
		# Grid over which we will calculate J
		extension_factor = 3.2
		theta0_maxplot = extension_factor * max( abs(self.theta_history[:, 0].min()), abs(self.theta_history[:, 0].max()) )
		theta1_maxplot = extension_factor * max( abs(self.theta_history[:, 1].min()), abs(self.theta_history[:, 1].max()) )
		theta0_vals = np.linspace(-theta0_maxplot, theta0_maxplot, 100)
		theta1_vals = np.linspace(-theta0_maxplot, theta0_maxplot, 100)

		# initialize J_vals to a matrix of 0's. Notice how we're passing a tuple to np.zeros, which contain the size (shape) of the zero-array we want to create
		J_values = np.zeros((theta0_vals.size, theta1_vals.size))

		# Fill out J_vals
		for i, theta0 in enumerate(theta0_vals):
			for j, theta1 in enumerate(theta1_vals):
				thetaT = np.array([theta0, theta1])
				J_values[i, j] = self.cost(self.X_train, self.y_train, thetaT)

		# Transpose to correct shape for contour plot
		J_values = J_values.T 

		A, B = np.meshgrid(theta0_vals, theta1_vals)
		C = J_values

		cp = plt.contourf(A, B, C)
		plt.colorbar(cp)
		plt.plot(self.theta_history[:, 0], self.theta_history[:, 1], 'r--')  # Gradient descent path
		plt.xlabel('Theta 0')
		plt.ylabel('Theta 1')
		plt.title('Contour plot of Cost Function with Gradient Descent Path')
		plt.show()

	def learning_curves(self):
		'''
		With learning curves you can have insights on the quality of the ML model you've trained, and diagnose your model
		(if it's prone to overfitting, underfit, etc)
		ACHTUNG: The learning curves are computed on the TRAIN SET and VALIDATION SET (since they are used to diagnose the model, and
		understand if it's generalizing the data, or only just "learning" them). They are not computed on the test set, 
		since if learning curves are not what we expect we should improove/change hypeparameters of our model (for example add other
		features). Only at the end, when we've choosen our final model through learning curves, we compute the performance metrics
		on the test set
		:return: two lists, cost_history_train, cost_history_valid; those are not "normal" cost history, in the sense that cost_history_train
		isn't equal to self.cost_history, which has n_steps entries, for every loop; instead cost_history_train and cost_history_valid represent
		the cost evolution of the training and validation set with incremental m (from 1 to m) samples during training phase. That means at first
		we train the algorithm with only one sample; then with two; then with three, etc.. until we arrive to consider all the m samples.
		'''
		m_train = len(self.X_train)
		m_values = np.linspace(2, m_train, 500, dtype=int)  # 10 points between 2 and m_train
		incremental_cost_train =[]
		incremental_cost_valid = []
		
		# Let's compute the final J(theta) (that is after n_steps iteration of gd) for progressively increasing number of samples, on the same dataset
		# The validation set is smaller than the training set, so we're going to consider a maximum sample-size of m_valid
		for m in m_values:
			X_train_m = self.X_train[:m]
			y_train_m = self.y_train[:m]
			# train the model using only the training set
			cost_history_train, theta_history_train = self.fit(X_train_m, y_train_m, update_internal=False)
			theta_trained_msamples = theta_history_train[-1]
			# Take the final cost (the one at the end of the n_step epoch of gd algorithm)
			incremental_cost_train.append(cost_history_train[-1])
			# Compute the cost on the validation training set
			incremental_cost_valid.append(self.cost(self.X_valid, self.y_valid, theta_trained_msamples))

		fig, ax = plt.subplots(figsize=(12,8))
		ax.set_xlabel('Sample size (m)')
		ax.set_ylabel('J(theta)')
		c, = ax.plot(m_values, incremental_cost_train, 'b.')
		cv, = ax.plot(m_values, incremental_cost_valid, 'r+')
		# ax.set_yscale('log')
		c.set_label('Training cost Jtr(theta)')
		cv.set_label('Validation cost Jcv(theta) computed on the theta obtained by training the model with m samples on the training subset')
		plt.title('Learning curves')
		ax.legend()
		plt.show()

	def animate(self):
		# Prepare the figure and subplots
		fig = plt.figure(figsize=(12, 5))

		# First subplot: regression line over training data
		ax1 = fig.add_subplot(121)
		ax1.plot(self.X_train[:, 1], self.y_train, 'ro', label='Training data')
		ax1.set_title('Housing Price Prediction')
		ax1.set_xlabel("Size of house in ft^2 (X1)")
		ax1.set_ylabel("Price in $1000s (Y)")
		ax1.grid(axis='both')
		ax1.legend(loc='lower right')

		line, = ax1.plot([], [], 'b-', label='Current Hypothesis')
		annotation = ax1.text(-2, 3, '', fontsize=20, color='green')
		annotation.set_animated(True)

		# Second subplot: contour plot of cost function
		ax2 = fig.add_subplot(122)
		
		# Generate contour plot for the cost function
		theta0_vals = np.linspace(self.theta_history[:, 0].min(), self.theta_history[:, 0].max(), 100)
		theta1_vals = np.linspace(self.theta_history[:, 1].min(), self.theta_history[:, 1].max(), 100)
		J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
		for i, theta0 in enumerate(theta0_vals):
			for j, theta1 in enumerate(theta1_vals):
				thetaT = np.array([theta0, theta1])
				J_vals[i, j] = self.cost(self.X_train, self.y_train, thetaT)
		J_vals = J_vals.T
		A, B = np.meshgrid(theta0_vals, theta1_vals)
		C = J_vals
		cp = ax2.contourf(A, B, C)
		plt.colorbar(cp, ax=ax2)
		ax2.set_title('Filled Contours Plot')
		ax2.set_xlabel('theta 0')
		ax2.set_ylabel('theta 1')

		track, = ax2.plot([], [], 'r-')
		point, = ax2.plot([], [], 'ro')

		# Initialize the plot elements
		def init():
			line.set_data([], [])
			track.set_data([], [])
			point.set_data([], [])
			annotation.set_text('')
			return line, track, point, annotation

		# Animation function that updates the frame
		def animate(i):
			# Update line for the regression prediction
			fit1_X = np.linspace(self.X_train[:, 1].min(), self.X_train[:, 1].max(), 1000)
			fit1_y = self.theta_history[i][0] + self.theta_history[i][1] * fit1_X

			# Update the gradient descent track
			fit2_X = self.theta_history[:i, 0]
			fit2_y = self.theta_history[:i, 1]

			# Set updated data for the plot elements
			track.set_data(fit2_X, fit2_y)
			line.set_data(fit1_X, fit1_y)
			point.set_data(self.theta_history[i, 0], self.theta_history[i, 1])

			# Update annotation with the current cost
			annotation.set_text(f'Cost = {self.cost_history[i]:.4f}')
			return line, track, point, annotation

		# Create the animation
		anim = animation.FuncAnimation(fig, animate, init_func=init,
									frames=len(self.cost_history), interval=50, blit=True)

		# Save animation as GIF
		anim.save('animation.gif', writer='imagemagick', fps=30)

		plt.close()  # Close the plot to prevent it from displaying statically
