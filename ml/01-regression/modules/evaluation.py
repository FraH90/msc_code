import numpy as np

np.random.seed(123)

class Evaluation:

    def __init__(self, model):
        self._model = model

    def compute_performance(self, X, y):
        """
        compute performance for linear regression model
        :param X: test sample with shape (m, n_features)
        :param y: ground truth (correct) target values shape (m,)
        :return: a dictionary with name of specific metric as key and specific performance as value
        """
        h_theta = self._model.predict(X)

        mae = self._mean_absolute_error(h_theta, y)
        mape = self._mean_absolute_percentage_error(h_theta, y)
        mpe = self._mean_percentage_error(h_theta, y)
        mse = self._mean_squared_error(h_theta, y)
        rmse = self._root_mean_squared_error(h_theta, y)
        r2 = self._r_2(h_theta, y)
        return {'mae': mae, 'mape': mape, 'mpe': mpe, 'mse': mse, 'rmse': rmse, 'r2': r2}

    def _mean_absolute_error(self, h_theta, y):
        """
        compute mean absolute error
        :param pred: prediction value with shape (m,)
        :param y: ground truth (correct) target values with shape (m,)
        :return: mean absolute error performance, MAE output is non-negative floating point. The best value is 0.0.

        Examples
        --------
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mean_absolute_error(y_true, y_pred)
        0.5
        """
        # Here output errors will be a vector; then we average it to obtain the mean absolute error
        output_errors = np.abs(h_theta - y)
        return np.average(output_errors)

    def _mean_squared_error(self, h_theta, y):
        """
        compute mean squared error
        :param pred: prediction value with shape (m,)
        :param y: ground truth (correct) target values with shape (m,)
        :return: mean squared error performance

        Examples
        --------
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mean_squared_error(y_true, y_pred)
        0.612...
        """
        # Same as mean absolute error, but we square all the errors here (remember **2 is elementwise operation)
        output_errors = (h_theta - y) ** 2
        return np.average(output_errors)

    def _root_mean_squared_error(self, h_theta, y):
        """
        compute root mean squared error
        :param pred: prediction value with shape (m,)
        :param y: ground truth (correct) target values with shape (m,)
        :return: root mean squared error performance

        Examples
        --------
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> root_mean_squared_error(y_true, y_pred)
        0.375
        """
        # Just the square root of the mean squared error
        return np.sqrt(self._mean_squared_error(h_theta, y))

    def _mean_absolute_percentage_error(self, h_theta, y):
        """
        compute mean absolute percentage error
        :param pred: prediction value with shape (m,)
        :param y: ground truth (correct) target values with shape (m,)
        :return: mean absolute percentage error (MAPE)

        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.

         Examples
        --------
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> mean_absolute_percentage_error(y_true, y_pred)
        0.3273...
        """
        output_errors = np.abs((h_theta - y) / y)
        return np.average(output_errors)

    def _mean_percentage_error(self, h_theta, y):
        """
        compute mean percentage error
        :param pred: prediction value with shape (m,)
        :param y: ground truth (correct) target values with shape (m,)
        :return: mean percentage error (MPE)
        """
        output_errors = (h_theta - y) / y
        return np.average(output_errors) * 100

    def _r_2(self, h_theta, y):
        """
        compute r2 score. R2 tell us how much the model is able to explain the variance in our data (lower is better?)
        :param pred: prediction value with shape (m,)
        :param y: ground truth (correct) target values with shape (m,)
        :return: r2 score

        Examples
        --------
        >>> y_true = [3, -0.5, 2, 7]
        >>> y_pred = [2.5, 0.0, 2, 8]
        >>> r_2(y_true, y_pred)
        0.948...
        """
        # See notes for this
        sst = np.sum((y - y.mean()) ** 2)
        ssr = np.sum((h_theta - y) ** 2)

        r2 = 1 - (ssr / sst)
        return r2 