import matplotlib.pyplot as plt
from modules.linear_regression import LinearRegression
from modules.evaluation import Evaluation
import os

plt.style.use(['ggplot'])

csv_file_relative_path = 'datasets\houses.csv'
csv_file = os.path.join(os.getcwd(), csv_file_relative_path)

def main():
    config = {'learning_rate': 1e-2, 'n_steps': 2000, 'features_select': 'GrLivArea, YearBuilt, MSSubClass, YearRemodAdd, FullBath, GarageArea', 'y_label': 'SalePrice', 'lmd': 1}
    linear_regression = LinearRegression(csv_file, config)
    linear_regression.fit()
    # Print results
    print(f"Thetas: {linear_regression.theta}")
    print(f"Final train cost/MSE (on training set) : {linear_regression.cost_history[-1]:.3f}")

    # Plot regression line
    linear_regression.plot_regression_line()
    
    # Plot cost history
    # You can modify this such that it plots together the cost history with respect to the training set and also wrt the validation set
    linear_regression.plot_cost_training_history()
    linear_regression.plot_3d_cost()
    linear_regression.gd_contour_plot()

    # Evaluation
    evaluation = Evaluation(linear_regression)
    performance = evaluation.compute_performance(linear_regression.X_test, linear_regression.y_test)
    for key, value in performance.items():
        print(f"{key} = {value}")

    # Learning curves
    linear_regression.learning_curves()

if __name__ == "__main__":
    main()

