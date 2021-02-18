"""Exact solution for a linear regression."""
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

w = (n * sum_xy - sum_y * sum_x) / (n * sum_x2 - sum_x**2)
b = (sum_y - w * sum_x) / n

y_est = [w * x + b for x in x_points]

x_points = pd.Series(x_points, name="X Variable")
y_points = pd.Series(y_points, name="Y Variable")

sns.regplot(x=x_points, y=y_points, ci=None)
plt.show()