import numpy as np

H = np.array([9, 15, 25, 14, 10, 18, 0, 16, 5, 19, 16, 20])
M = np.array([39, 56, 93, 61, 50, 75, 32, 85, 42, 70, 66, 80])

print 'Hours total: {0}, average: {1:.2f}, std: {2:.2f}'.format(H.sum(), H.mean(), H.std())
print 'Marks total: {0}, average: {1:.2f}, std: {2:.2f}'.format(M.sum(), M.mean(), M.std())

print 'Hours variance: {0:.2f}'.format(H.var())
print 'Marks variance: {0:.2f}'.format(M.var())

cov_matrix = np.cov(H, M, bias=1)
print type(cov_matrix)

print 'Covariance: {0}'.format(cov_matrix)
