"""Linear regression example."""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

x_points = [1.1, 1.5, 2.5, 3.2]
y_points = [3.3, 4.5, 6.0, 9.0]

sum_x = sum(x_points)
sum_y = sum(y_points)
sum_x2 = sum([x * x for x in x_points])
sum_xy = sum([x * y for x, y in zip(x_points, y_points)])

a = (sum_xy - sum_y * sum_x) / (sum_x2 - sum_x * sum_x)
b = sum_y - a * sum_x

y_est = [a * x + b for x in x_points]

x_points = pd.Series(x_points, name="X Variable")
y_points = pd.Series(y_points, name="Y Variable")

sns.regplot(x=x_points, y=y_points, ci=None)
plt.show()
