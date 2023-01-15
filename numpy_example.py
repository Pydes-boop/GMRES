from gmres_numpy.GMRES import *
from gmres_numpy.ABGMRES import *
from gmres_numpy.BAGMRES import *

#### Simple symmetric example ####
nmax_iter = 5
epsilon = None
A = np.matrix('1 1; 3 -4')
b = np.array([3, 2])
x0 = np.array([0, 0])
solution = np.array([2., 1.])

x, r = GMRES(A, b, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("GMRES: ")
print(" x: ", x, " should be: ", solution)

x, r = ABGMRES(A, A.T, b, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("ABGMRES: ")
print(" x: ", x, " should be: ", solution)

x, r = BAGMRES(A, A.T, b, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("BAGMRES: ")
print(" x: ", x, " should be: ", solution)
print("\t---\t")

#### Unsymmetric Example ####
#### Dont overiterate ####

nmax_iter = 3
epsilon = None
A = np.matrix('1 2 3 4;1 2 3 -4;1 -2 3 4')
b = np.array([3, 2, 1])
x0 = np.array([0, 0, 0, 0])
solution = np.array([0.15 , 0.5  , 0.45 , 0.125])

x, r = GMRES(np.dot(A.T, A), np.dot(A.T, b), x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("GMRES: ")
print(" x: ", x, " should be: ", solution)

x, r = ABGMRES(A, A.T, b, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("ABGMRES: ")
print(" x: ", x, " should be: ", solution)

x, r = BAGMRES(A, A.T, b, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("BAGMRES: ")
print(" x: ", x, " should be: ", solution)
print("\t---\t")