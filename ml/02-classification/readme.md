# Example of univariate linear regression

## Houses univariate
a_houses_univariate.py
This train a regressor on the dataset houses_portland_simple.csv. It uses the 80/20 splitting rule (80% training dataset, 20% test set); then the 30% of the training set is retained as validation set 

## Important
- ID: sample number. There are 1440 entries, so 1440 samples
- All other columns are features (x1, x2, ... features column vectors) except the final column, which is the label (y column vector, 'SalePrice')
- We will train a regression algorithm based on the dataset given in the csv file
- Final goal: given a certain set of value for the features (inputs), we want to predict the value at which the house with that features will be sold


- We'll use pandas to read the csv file and import it into a pandas dataframe
- With the describe() pandas method we can have basic statistics info on the dataset (ex mean, standard deviation, 25-50-75% percentile/quartiles, etc)

- We can select a subset of the features to start with (GrLivingArea, LotArea, GarageArea, FullBath). This selection is just "random" (we're still not deciding which feature are really relevant).
- From the pandas dataframe, generate a numpy array (remember, pandas is not for computation, pandas it is!)