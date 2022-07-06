import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

"""
I used the following resources to help me complete this task: 
- for understanding the diabetes dataset:
    (https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt)
- for the steps needed:
    (https://scikit-learn.org/stable/auto_examples/linear_model/
    plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py)
- for calculating the Euclidean distance:
    (https://towardsdatascience.com/
    linear-regression-the-actually-complete-introduction-67152323fcf2)
"""


# load the dataset
d_set = load_diabetes()

# get the data of one column (BMI standardized with 0 as the mean)
# leave out 'np.newaxis' to keep the array one-dimensional
bmi_values = d_set.data[:,2]

# divide the data into a training set
x_train = bmi_values[:-20]
# get the response variable values
y_train = d_set.target[:-20]

# get the last 20 values for tests
x_test = bmi_values[-20:]
y_test = d_set.target[-20:]

# a method to return the gradient and intercept of y = mx + b
def find_line(x_array, y_array):
    # use the formula 𝑚 = (μ(𝑥) * μ(𝑦) − μ(𝑥 * 𝑦))/((μ(𝑥))2 − μ(𝑥2))
    m = ((np.mean(x_array) * np.mean(y_array)) - np.mean(x_array * y_array)) / \
        (np.mean(x_array)**2 - np.mean(x_array**2))
    # calculate the b using 𝑏 = μ(𝑦) − 𝑚 * μ(𝑥)
    b = np.mean(y_array) - m * np.mean(x_array)
    # return the slope (m * x_array) + b
    return m, b


# use training data to find the m and b values of the best fit line
# m is the first element returned
trained_m = find_line(x_train, y_train)[0]
# b is the second element
trained_b = find_line(x_train, y_train)[1]

# create the graph
# scatter plot of the training data
plt.scatter(x_train, y_train, c='r')
# scatter plot of the test data
plt.scatter(x_test, y_test, c='g')
"""for the y values of the testing array use y = mx + b where m and b are 
trained_m and trained_b as calculated above"""
plt.plot(x_test, ((trained_m * x_test) + trained_b), c='b')
# add the legend
plt.legend(["Training Data", "Testing Data", "Line of Best-fit"])
plt.show()



