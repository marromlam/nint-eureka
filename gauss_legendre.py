# mauroandrea
#
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["marromlam@gmail"]


from string import Template
import numpy as np
# import pyopencl as cl
import ipanema
ipanema.initialize('opencl', 1)


# integration configuration {{{

omega_min = np.float64(2*np.pi)
omega_max = np.float64(20*np.pi)
omega_min = np.float64(0.5)
omega_max = np.float64(1)
number_of_points = np.int32(100)

# }}}


# kernel {{{

kernel_src = Template(open('mauroandrea.c').read())
kernel_src = kernel_src.substitute(dict(
  nknots=number_of_points+1,
))
# prg = cl.Program(ctx, kernel_src).build()
prg = ipanema.compile(kernel_src)
gaussleg = prg.kernel_gauss_legendre
print("Compilation :: Done!")



# }}}


# tester for knots computations {{{
# since first element is 0, we need to add 1 to the number of points
k = ipanema.ristra.allocate(np.zeros((number_of_points+1)))
w = ipanema.ristra.allocate(np.zeros((number_of_points+1)))

gaussleg(omega_min, omega_max, number_of_points, k, w, global_size=(1,), local_size=(1,))
print("Execution :: Done!")

print("  knots: ", k)
print("weights: ", w)

# }}}


# Get the job done {{{

print("\n\nACTUAL MAGIG\n")

gauss_quadrature = prg.qgaus
ans = ipanema.ristra.allocate(np.float64([0]))
gauss_quadrature(ans, omega_min, omega_max, global_size=(1,), local_size=(1,))
print(ans)
print("Execution :: Done!")

# }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
