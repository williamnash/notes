"""Solution for a linear regression using gradient descent."""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

x_points = [1.1, 1.5, 2.5, 3.2]
y_points = [3.3, 4.5, 6.0, 9.0]

n      = len(x_points)
sum_x  = sum(x_points)
sum_y  = sum(y_points)
sum_x2 = sum([x * x for x in x_points])
sum_xy = sum([x * y for x, y in zip(x_points, y_points)])

# hyperparameters
n_epochs = 10000
alpha = 0.001

# initialization
w = 0
b = 0

def update_weights(w, b, alpha):
    """Updates the weights according to the gradient"""

    delta_w = 2 * (w * sum_x2 + b * sum_x - sum_xy)
    delta_b = 2 * (b * n + w * sum_x - sum_y)

    w -= alpha * delta_w * w
    b -= alpha * delta_b * b

    return w, b

for epoch in range(n_epochs):
    w, b = update_weights(w, b, alpha)

y_est = [w * x + b for x in x_points]

x_points = pd.Series(x_points, name="X Variable")
y_points = pd.Series(y_points, name="Y Variable")

sns.regplot(x=x_points, y=y_points, ci=None)
plt.show()