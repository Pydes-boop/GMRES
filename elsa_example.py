from gmres_elsa import *

nmax_iter = 10
epsilon = None

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
projector = elsa.JosephsMethodCUDA(volume_descriptor, sino_descriptor, False)

# simulate the sinogram
sinogram = projector.apply(phantom)

# new backprojection
backprojector = elsa.adjoint(projector)

# baue datacontainer mit 0 gefüllt welcher der größe und form des phantoms entsprechen
x0 = elsa.DataContainer(np.zeros_like(np.asarray(phantom)), phantom.getDataDescriptor())

# cant apply(B, A) with elsa so we cant test normal GMRES with elsa

x, r = ABGMRES(projector, backprojector, sinogram, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("ABGMRES: ")
print("Difference to true phantom: ", np.linalg.norm(np.subtract(phantom, np.asarray(x))))

x, r = BAGMRES(projector, backprojector, sinogram, x0, nmax_iter, epsilon=epsilon)
print("\t---\t")
print("BAGMRES: ")
print("Difference to true phantom: ", np.linalg.norm(np.subtract(phantom, np.asarray(x))))
print("\t---\t")