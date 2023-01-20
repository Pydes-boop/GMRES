# GMRES for unmatched Matrices/Projectors A and B

Implementation of GMRES, AB-GMRES and BA-GMRES for unmatched projector/backprojector pairs or Matrices in Python using NumPy. Alternatively using python bindings for elsa tomographic reconstruction library. Basis for elsas GMRES implementations you can find in the [elsa repository](https://gitlab.lrz.de/IP/elsa).

## General Idea

In this repository 3 different algorithms are implemented. First the basic [GMRES](gmres_numpy/GMRES.py) which solves indefinite, nonsymmetric, square system $Ax = b$. This is a basic Krylov-Subspace solver, which can be expanded by introducing the following expansions to our System $Ax = b$:
\
\
$\min_{z \epsilon \mathbb{R}^{N}} \|b-ABz\|$\
$\min_{x \epsilon \mathbb{R}^{M}} \|Bb-BAx\|$\
$x = Bz$\
\
These so-called unmatched normal equations can be solved via [AB-GMRES](gmres_numpy/ABGMRES.py) and [BA-GMRES](gmres_numpy/BAGMRES.py). The specific use-cases can be different but it can be benefitial in any case taking advantage of an unmatched transpose $B \neq A^T$.

## Application in computer tomography

For computer tomography it can be benefitial to use such an unmatched transpose as a unmatched projector / backprojector pair. This has a multitude of applications but is typically interesting as different methods for forward and backward projection can be used, or the unmatched backprojector can be used as a preconditioner.

## Performance

The GMRES methods show a slighty slower speed compared to solvers like Conjugate Gradient (CG), but they should really be used in cases where CG does not make sense or is not applicable.

## Using with elsa - an elegant framework for tomographic reconstruction

Elsa is an open source framework for tomographic reconstruction utilizing features from modern CPP for perfomant results. It can be installed from [here](https://gitlab.lrz.de/IP/elsa) and provides alternatives for solvers as well as a faster GMRES implementation.
You can find current installation instructions for the python bindings which we will be using in the elsa documentation [https://ciip.in.tum.de/elsadocs/](https://ciip.in.tum.de/elsadocs/guides/python_guide/install.html). 

```
# Prerequisites
- C++17 compliant compiler: GCC >= 9 or Clang >= 9
- CMake >= 3.14
- CUDA toolkit >= 10.2
```

```bash
# install python bindings for elsa

pip install numpy matplotlib scipy

git clone https://gitlab.lrz.de/IP/elsa
cd elsa

# using the --verbose flag the console output should show you if CUDA is enabled

pip install . --verbose

# sometimes this has difficulty to find CUDAs thrust on your system, try linking your CUDA directory for CMake:

CMAKE_ARGS="-DThrust_DIR=/usr/local/cuda-11.8/targets/x86_64-linux/lib/cmake/thrust" pip install . --verbose
```

Using elsas python bindings to generate a problem is simple, the source and a further explanation can be found [here](https://ciip.in.tum.de/elsadocs/guides/python_guide/forward_projection.html).

```python
import pyelsa as elsa

nmax_iter = 10
epsilon = None

# Phantom Size
size = np.array([32, 32])
phantom = elsa.phantoms.modifiedSheppLogan(size)
volume_descriptor = phantom.getDataDescriptor()

# settings
num_angles = 30
arc = 180

# generate circular trajectory
sino_descriptor = elsa.CircleTrajectoryGenerator.createTrajectory(
    num_angles, phantom.getDataDescriptor(), arc, size[0] * 100, size[0])

# setup operator for 2d X-ray transform
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor)

# simulate the sinogram
sinogram = projector.apply(phantom)

# new unmatched backprojection (because of JosephsMethodCUDA)
backprojector = elsa.adjoint(projector)
```

This Problem can then be solved using the provided elsa [AB-GMRES](gmres_elsa/ABGMRES_elsa.py) and [BA-GMRES](gmres_elsa/ABGMRES_elsa.py) solvers as shown in the [elsa_example.py](elsa_example.py), where x is the reconstructed image and r is the residual:

```python
x, r = ABGMRES(projector, backprojector, sinogram, x0, nmax_iter, epsilon=epsilon)
x, r = BAGMRES(projector, backprojector, sinogram, x0, nmax_iter, epsilon=epsilon)
```

they can still be used for any NumPy example as they define the typical Matrix Vector Product with its own function, differentiating between elsa DataContainers and NumPy / Python containers:

```python
def apply(A, x):
    if isinstance(A,(list,np.ndarray, np.matrix)):
        return np.asarray(np.dot(A, x)).reshape(-1)
    else:
        return np.asarray(A.apply(elsa.DataContainer(x)))
```


## References

### Links
- [elsa repository](https://gitlab.lrz.de/IP/elsa)
- [elsa documentation](https://ciip.in.tum.de/elsadocs/)

### Papers
- [GMRES Methods for Tomographic Reconstruction with an Unmatched Back Projector](http://arxiv.org/abs/2110.01481)
- [GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems](http://epubs.siam.org/doi/10.1137/0907058)
- [GMRES Methods for Least Squares Problems](http://epubs.siam.org/doi/10.1137/070696313)
     
