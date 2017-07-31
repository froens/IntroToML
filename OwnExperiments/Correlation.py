from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import numpy


def f(x):
    alpha = 2
    beta = 5
    return alpha*x+beta


X = range(0, 100)
y = [f(n) for n in X]

# Add noise
Y = numpy.random.normal(y, 150)

cor, _ = pearsonr(X, Y)
print cor

plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X, Y, c="r", alpha=0.5)
plt.show()