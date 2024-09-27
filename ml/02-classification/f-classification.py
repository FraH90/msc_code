import matplotlib.pyplot as plt
from modules.logistic_regression import LogisticRegression
from modules.classification_metrics import ClassificationMetrics
import os

plt.style.use(['ggplot'])

csv_file_relative_path = 'datasets\diabetes.csv'
csv_file = os.path.join(os.getcwd(), csv_file_relative_path)

def main():
    config = {'learning_rate': 1e-2, 'n_steps': 2000, 'features_select': 'Glucose', 'class': 'Outcome', 'lmd': 1}
    logistic_regression = LogisticRegression(csv_file, config)
    logistic_regression.fit()
    # Print results
    print(f"Thetas: {logistic_regression.theta}")
    print(f"Final train cost/MSE (on training set) : {logistic_regression.cost_history[-1]:.3f}")

    # Plot something (correspective of linear regr?)
    """ 
    # Plot cost history
    logistic_regression.plot_cost_training_history()
    logistic_regression.plot_3d_cost()
    logistic_regression.gd_contour_plot()
    """

    # Evaluation
    y_predicted = logistic_regression.predict(logistic_regression.X_test, logistic_regression.theta)
    metrics = ClassificationMetrics(logistic_regression.y_test, y_predicted)
    performance = metrics.compute_errors()
    for key, value in performance.items():
        print(f"{key} = {value}")

if __name__ == "__main__":
    main()

