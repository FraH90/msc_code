import matplotlib.pyplot as plt
from modules.linear_regression import LinearRegression
from modules.evaluation import Evaluation
import os

plt.style.use(['ggplot'])


def main():
    csv_file_relative_path = 'datasets\Position_Salaries_base.csv'
    csv_file = os.path.join(os.getcwd(), csv_file_relative_path)

    config = {'learning_rate': 1e-2, 'n_steps': 2000, 'features_select': 'Level', 'poly_grade': '1, 2, 3', 'y_label': 'Salary', 'lmd': 1}
    linear_regression = LinearRegression(csv_file, config)
    linear_regression.fit()
    # Print results
    print(f"Thetas: {linear_regression.theta}")
    print(f"Final train cost/MSE (on training set) : {linear_regression.cost_history[-1]:.3f}")

    # Plot regression line
    linear_regression.plot_regression_poly()
    
    # Plot cost history
    # You can modify this such that it plots together the cost history with respect to the training set and also wrt the validation set
    linear_regression.plot_cost_training_history()

    # Evaluation
    evaluation = Evaluation(linear_regression)
    performance = evaluation.compute_performance(linear_regression.X_test, linear_regression.y_test)
    for key, value in performance.items():
        print(f"{key} = {value}")

if __name__ == "__main__":
    main()

