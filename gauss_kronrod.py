# gauss_kronrod
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import numpy as np
import ipanema
ipanema.initialize('opencl', 1, real='float')

prog = ipanema.compile(open('gauss_kronrod.c').read())

a = np.float32(-0.0)
b = np.float32(+1.0)
integral = ipanema.ristra.allocate(np.float32([0.]))
prog.kernel_quadgk(integral, a, b, global_size=(1,), local_size=(1,))
print("Integral:", integral)


# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
