import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import operator

DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_N_STEPS = 2000
DEFAULT_LMD = 1

np.random.seed(123)

class LinearRegression:
	"""
	:param csv_path: path of the csv file containing the data we want to fit
	:param config: dictionary that contain the configuration of the linear regressor (hyperparameters, convergence condition, feature selection)
	Class to perform learning for a linear regression. It has all methods to be trained with different strategies
	and one method to produce a full prediction based on input samples (inference). 
	This is completed by the class Evaluation in the module evaluation.py that measure performances and various indicators.
	"""
	def __init__(self, csv_path, config= {'learning_rate': 1e-2, 'n_steps': 2000, 'features_select': 'Size', 'poly_grade': '', 'y_label': 'Price', 'lmd': 1} ):
		"""
		:param learning_rate: learning rate value
		:param n_steps: number of epochs around gradient descent
		:param n_features: number of features involved in regression
		:param lmd: regularization factor (lambda)
		"""
		# Parameters
		self.csv_path = csv_path
		self.config = config
		self.learning_rate = config['learning_rate'] if config['learning_rate'] != '' else DEFAULT_LEARNING_RATE
		self.n_steps = int(config['n_steps']) if config['n_steps'] != '' else DEFAULT_N_STEPS
		self.features_select = config['features_select']
		self.y_label = config['y_label']
		self.lmd = config['lmd'] if 'lmd' in config else DEFAULT_LMD

		# Placeholders
		self.X, self.y = None, None
		self.X_train, self.y_train = None, None
		self.X_valid, self.y_valid = None, None
		self.X_test, self.y_test = None, None
		self.theta = None
		self.cost_history = None
		self.theta_history = None
		# Those will describe the mean and stddev of each feature (column) of the dataset
		self.mean_trainingset, self.std_trainingset  = None, None

		# SEMI-AUTOMATIC BEHAVIOR
		# Automatically load data, perform preprocessing, etc
		self._load_and_preprocess()
		# Generate eventual polynomial features
		self._polynomial_features()
		# get m and n_features
		self.m_samples = self.X.shape[0]

		# split dataset
		self._dataset_split()
		# normalize data
		self._normalize()
		# add bias column
		self._add_bias_column()
		# update n_features after adding bias column (and eventual polynomial features)
		self.n_features = self.X_train.shape[1]
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
		# read the dataset of houses prices
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

	def _polynomial_features(self):
		'''
		Add polynomial features, given by the poly_grade field of config dictionary. Permitted only if the initial dataset has a single feature.
		'''
		if self.n_features != 2:
			print("Polynomial features can be retrieven only in case of single feature (X must be a vector, not a matrix)")
			print("Simple multivariate regression will be performed here, without taking into account poly features")
			return
		# See if there are polynomial features (x^2, x^3, etc)
		if self.config['poly_grade'] == '' or self.config['poly_grade'] == '1':
			self.poly_grade = [1]
			return
		else:
			# Obtain from poly_feature string the grades for which we need to do regression
			polygrade_string_splitted = str(self.config['poly_grade']).split(", ")
			self.poly_grade = [int(exp_string) for exp_string in polygrade_string_splitted]
		# Being the input univariate, we just need to extract the single feature from the dataset and compute a new self.X matrix from x
		x = self.X[:, 0]
		col_stack = [x**exp for exp in self.poly_grade]
		self.X = np.column_stack(col_stack)

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
		self.mean_trainingset = self.X_train.mean(axis=0)
		self.std_trainingset = self.X_train.std(axis=0)
		# z-score normalization. Achtung: mean has another shape (it's a row), but by broadcasting (implicit operation)
		# In broadcasting the array "mean" here is replicated the same number of rows of X_train in order to let the arithmetic operation happen
		self.X_train = (self.X_train - self.mean_trainingset) / self.std_trainingset
		# Apply the same normalization to validation and test sets. THIS IS IMPORTANT! OTHERWISE PERFORMANCE METRICS WILL BE WRONG!
		# NOTICE HOW TO NORMALIZE THE FEATURES OF VALIDATION AND TEST SET WE USE THE MEAN AND STDDEV OF THE TRAINING SET
		self.X_valid = (self.X_valid - self.mean_trainingset) / self.std_trainingset
		self.X_test = (self.X_test - self.mean_trainingset) / self.std_trainingset


	def _add_bias_column(self):
		# Add bias column (of ones) to X_train, X_valid, X_test. Remember this is needed when computing h_theta(x), so theta0 is included in the summation
		# This ensure we can compute h_theta(x) on all the datasets by simply using a matrix product
		# ACHTUNG: the bias column must be added AFTER normalization
		# The c_ attribute of a nparray is to create a nparray as concatenation of columns,
		# so we generate a nparray of ones of the same dimension of X_train, and then we concatenate this to the original X_train
		self.X_train = np.c_[np.ones(self.X_train.shape[0]), self.X_train]
		self.X_valid = np.c_[np.ones(self.X_valid.shape[0]), self.X_valid]
		self.X_test = np.c_[np.ones(self.X_test.shape[0]), self.X_test]

	def predict(self, X):
		"""
		Perform a complete prediction about X samples (that is, it computes h_theta(X))
		OSS: The prediction expression doesn't change if we compute thetas with or without regularization
		:param X: test sample with shape (m, n_features)
		:return: prediction with respect to X sample. The shape of return array is (m, )
		"""
		return np.dot(X, self.theta)
	
	def cost(self, X, y, theta):
		m = len(X)
		h_theta = np.dot(X, theta)
		error = h_theta - y
		# The regularization part of the cost is the squared module of theta, multiplied by lambda, with the exception of the 0-th element
		# which is not regularized (so this means for i=0, in the scalar equation, the regularization term is not present in the cost expression)
		# To obtain that equivalent of the scalar equation in the vector form, we just perform dot product of theta.T and theta, but the last theta 
		# is multiplied element-wise (lmd_vector*theta) by the elements in lmd_vector, so we obtain that all the components squared of theta are summed up
		# and multiplied by lmd, with the exception of the 0-th component that is excluded from this computation (since lmd_vector[0] is null)
		# OSS: COST WILL BE HIGH IF Y IS HIGH, SINCE WE'VE NOT NORMALIZED THE LABELS Y
		cost =   1/(2*m) * ( np.dot(error.T, error) +  np.dot(theta.T, self.lmd_vector*theta) )
		return cost

	def fit(self, X=None, y=None, update_internal=True):
		"""
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
		if X is None:
			X = self.X_train
		if y is None:
			y = self.y_train

		# Initialize theta to a random value (uniform distribution, range 0-1)
		theta = np.random.rand(self.n_features)
		# cost_history, theta_history are just for plots purposes (they will be J(theta) and theta at every iteration), not needed for learning 
		cost_history = np.zeros(self.n_steps)
		theta_history = np.zeros((self.n_steps, self.n_features)) 

		# Get the number of samples of the dataset (that is, the number of rows of the matrix X). 
		# The first dimension of the matrix is always row, so len(X) returns number of rows!
		# OSS: If you want to get the number of columns (that is, the number of feature), you can use len(X[0])
		m = len(X)      
						
		# Here you should generate random parameters (a random vector theta of n_features elements) in order to init the theta vector,
		# but this has already been done in the __init__ method

		for step in range(0, self.n_steps):
			# Compute the hypothesis h_theta(x), for the current values of theta (random in the first run) and for all the samples in the training dataset
			# With a simple matrix product we'll get the predictons for all the samples (so for m sets of n features)
			# OSS: EVERY ITERATION OF THIS LOOP WILL BE AN EPOCH. SO THIS TRAINING IS CONSTITUTED OF self.n_steps EPOCHS.
			# NOTICE HOW BATCH GD USES THE ENTIRE DATASET, AT THE SAME TIME, TO COMPUTE THE PARAMETERS THETA, AND THE PARAMETERS ARE UPDATED ALL AT ONCE
			h_theta = np.dot(X, theta)
			# Compute the errors (prediction - dataset label value). This will be a vector, for all the samples
			# NOTICE HOW VECTOR OPERATIONS LET YOU DO EVERYTHING REAL QUICK! NO NEED TO WRITE LOOPS HERE TO CYCLE ALL THE SAMPLES!
			error = h_theta - y
			# Compute the gradient, with regularization (to do it without regularization it's sufficient to put lmd=0)
			# ANOTHER APPROACH HERE WOULD BE FOR THE REGULARIZATION TERM BE EQUAL TO (1/m)*(THETA.T * LMD_VEC), WHERE LMD_VEC IS A VECTOR CONTAINING 
			# ALL LAMBDA, EXCEPT THE 0-TH ELEMENT WHICH IS SET TO ZERO
			# ACHTUNG: THE PREDICTION IS NOT MODIFIED BY THE REGULARIZATION! THIS IS ONLY TO GIVE MORE STABILITY TO THE GRADIENT DESCENT
			gradient = (1/m) * ( np.dot(X.T, error) + (theta.T * self.lmd_vector) )
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
			cost_history[step] = 1/(2*m) * ( np.dot(error.T, error) +  np.dot(theta.T, self.lmd_vector*theta) )

		# Update the internal parameter of the model only if update_internal==True, otherwise just return the cost_history and theta_history
		if update_internal:
			self.theta = theta
			self.cost_history = cost_history
			self.theta_history = theta_history

		return cost_history, theta_history

	def fit_minibatch_gd(self, batch_size=10, update_internal=True):
		'''
		Fit the training dataset (that is, generate the parameters theta) by employing a minibatch gradient descent (compromise between batch gd and stochastic gd)
		If the size of the batch is batch_size, we can think of the minibatch gd as number_batches = m_samples/batch_size rounds of batch gd (while the scan of single samples
		inside a single batch can be seen as a pure stochastic gd, where at each sample we update the parameters and the next sample is chosen randomly from the one available
		in the current batch)
		:return: theta_history and cost_history
		'''
		cost_history_train = np.zeros(self.n_steps)
		cost_history_valid = np.zeros(self.n_steps)

		# Initialize theta to a random value (uniform distribution, range 0-1)
		theta = np.random.rand(self.n_features)
		# Initialize theta history (zero fill)
		theta_history = np.zeros((self.n_steps, theta.shape[0]))
		
		# Running through epochs
		for step in range(0, self.n_steps):
			cost = 0
			h_theta_valid = np.dot(self.X_valid, theta)
			error_valid = h_theta_valid - self.y_valid
			# Iterate through the various batches. Here the index i is the starting point of the current batch; i+batch_size-1 is the end of the batch
			for i in range(0, self.m_samples, batch_size):
				# Select the portion to create a batch from X_train, by slicing from i to i+batch_size (ex from 0 to 9, then from 10 to 19, then from 20 to 29...)
				X_i = self.X_train[i : i+batch_size]
				y_i = self.y_train[i : i+batch_size]
				h_theta = np.dot(X_i, theta)
				error = h_theta - y_i
				# Each one of those iteration through a single batch, is like a small batch gd; but the size is not m_samples, is batch_size, so in formula u have 1/batch_size
				gradient = (1/batch_size) * ( np.dot(X_i.T, error) + (theta.T * self.lmd_vector) )
				theta = theta - self.learning_rate * gradient
				cost += 1/(2*batch_size) * np.dot(error.T, error)
			# Here current epoch (step) has finished running, so put the results in the history lists
			theta_history[step, :] = theta.T
			cost_history_train[step] = cost
			cost_history_valid[step] = (1/(2*self.m_samples)) * np.dot(error_valid.T, error_valid)
		
		if update_internal==True:
			self.theta = theta
			self.cost_history = cost_history_train
			self.theta_history = theta_history

		return cost_history_train, cost_history_valid, theta_history

	def denormalize_thetas(self, theta):
		'''
		theta: learned parameter vector (including theta_0 for bias)
		mean: mean values of each feature
		std: standard deviation values of each feature
		Returns:
			theta_denorm: de-normalized thetas
		'''
		#################################################################################################################################
		# The theta vector computed on the normalized training set, correspond to the normalized feature space (that is, if you want to compute
		# the hypothesis on unseen data you need to use as features normalized ones!)
		# If you wish to plot the regression line over the original, unnormalized features, you need to denormalize theta so you can apply 
		# h_theta(x) to the original feature space
		#################################################################################################################################
		mean = self.mean_trainingset
		std = self.std_trainingset
		# Initialize the de-normalized theta vector
		theta_denorm = np.zeros_like(theta)
		# Update theta_1 to theta_n (features)
		theta_denorm[1:] = theta[1:] / std
		# Update theta_0 (intercept/bias term)
		theta_denorm[0] = theta[0] - np.sum((theta[1:] * mean) / std)
		return theta_denorm

	def plot_regression_line(self):
		'''
		This plot the regression line, only over the first feature x1 (indexed by X[:,1] since X[:,0] is the bias column)
		If other features are present, only the first feature is plotted with this method
		'''
		plt.figure(figsize=(10,6))
		# Scatter plot of training data points
		plt.scatter(self.X_train[:,1], self.y_train, color='r', label='Data points')
		# Plot the line, by generating 100 points in the range min-max of the training dataset, and computing the prediction h_theta on those points
		lineX = np.linspace(self.X_train[:,1].min(), self.X_train[:,1].max(), 100)
		liney = self.theta[0] + self.theta[1]*lineX
		plt.plot(lineX, liney, 'b--', label='Current hypothesis')
		# labels, title, legend
		plt.xlabel(self.features_list[0])
		plt.ylabel(self.y_label)
		plt.title(f'Regression line over {self.features_list[0]}')
		plt.legend()
		plt.show()

	def plot_regression_poly(self):
		plt.figure(figsize=(10,6))
		# Extract the single feature; it will be in column 1, since in col0 we have the bias
		x = self.X_train[:, 1]
		y_pred = self.predict(self.X_train)
		# Scatter plot of training data points
		plt.scatter(x, self.y_train, color='r', label='Data points')
		# Plot the line, by generating 100 points in the range min-max of the training dataset, and computing the prediction h_theta on those points
		sort_axis = operator.itemgetter(0)
		sorted_zip = sorted( zip(x, y_pred), key=sort_axis )
		x_poly, y_poly_pred = zip(*sorted_zip)
		plt.plot(x_poly, y_poly_pred, 'b--', label='Current hypothesis')
		# labels, title, legend
		plt.xlabel(self.features_list[0])
		plt.ylabel(self.y_label)
		plt.title(f'Regression line over {self.features_list[0]}')
		plt.legend()
		plt.show()

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
		for t1, element in enumerate(theta0_vals):
			for t2, element2 in enumerate(theta1_vals):
				thetaT = np.array([element, element2])
				J_values[t1, t2] = self.cost(self.X_train, self.y_train, thetaT)

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
		for t1, element in enumerate(theta0_vals):
			for t2, element2 in enumerate(theta1_vals):
				thetaT = np.array([element, element2])
				J_vals[t1, t2] = self.cost(self.X_train, self.y_train, thetaT)
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
