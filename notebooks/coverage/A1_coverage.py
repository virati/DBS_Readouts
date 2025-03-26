"""
[Paper] Readout Limitations: RvA and Coverage Metrics
Summary: This notebook focuses on introducing the concept of coverage as the inner product between $\mathbf{H}$ and $\mathbf{\Gamma}$

"""
# %%
import statsmodels.api as sm
from scipy.stats import pearsonr
from dbread.ro_sys import RO_SYS
import jax.numpy as np
import numpy as nnp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%
B = 1
M = 1
regions = 10

basic_system = RO_SYS(B=B, M=M, regions=regions)

basic_system.H = nnp.zeros((basic_system.regions, basic_system.B))
basic_system.H[4:, :] = 1
basic_system.gamma = nnp.zeros((basic_system.regions, basic_system.M))
basic_system.gamma[0:5, :] = 0.3


print(f"Coverage is: {str(basic_system.coverage()[0][0])}")

# %%
# we need a signal for the underlying regions
T = 10000
X_regions = nnp.random.multivariate_normal(
    0*np.zeros(regions), nnp.eye(regions), size=(T,))

plt.plot(X_regions)
plt.show()

# %%

predicted = nnp.dot(basic_system.H.T, X_regions.T).squeeze()
actual = nnp.dot(basic_system.gamma.T, X_regions.T).squeeze()

fig = plt.figure()
ax = fig.add_subplot(111)

X = predicted
Y = actual

plt.scatter(X, Y)
print(f"Pearson Corr {pearsonr(Y, X)}")
ax.set_aspect('equal')


model = sm.OLS(Y, X)
corr = model.fit()
print(f"Linear Corr {corr.params}")
print(f"Sqrt of slope {np.sqrt(corr.params)}")

lrmodel = LinearRegression()
lrmodel.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
rsquared = lrmodel.score(X.reshape(-1, 1), Y.reshape(-1, 1))
print(f"R Squared: {rsquared}")


# %%
X = actual
Y = predicted


fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(X, Y)
ax.set_aspect(aspect='equal')
print(pearsonr(actual, predicted))

model = sm.OLS(Y, X)
corr = model.fit()
print(corr.params)
