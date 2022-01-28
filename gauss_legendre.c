#ifndef _MAUROANDREA_C_
#define _MAUROANDREA_C_


#define USE_DOUBLE 1
#include <lib99ocl/core.c>

#define QUADRATURE_EPS 1e-14

KERNEL
void kernel_gauss_legendre(ftype x1, ftype x2, int n, __global ftype *x, __global ftype *w)
{
    // Given the lower and upper limits of integration x1 and x2, and given n,
		// this routine returns arrays x[1..n] and w[1..n] of length n, containing
		// the abscissas and weights of the Gauss- Legendre n-point quadrature
		// formula.
	  // printf("n = %d\n", n);
    ftype z1,pp,p3,p2,p1;
    int m = (n+1)/2;
    ftype xm = 0.5 * (x2+x1);
    ftype xl = 0.5 * (x2-x1);

		ftype z = 0;
    for (int i=1; i<=m; i++) {
	      // printf("i = %d\n", i);
        z = cos( 3.1415926 * (i-0.25)/(n+0.5) );
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j=1; j<=n; j++) {
                p3 = p2;
                p2 = p1;
								p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
						}
            pp = n*(z*p1-p2)/(z*z-1.0);
            z1 = z;
            z  = z1-p1/pp;  // newton
        } while (fabs(z-z1) > QUADRATURE_EPS);

		    // scale to the desired interval
	      // printf("n+1-i = %d\n", n+1-i);
        x[i]     = xm - xl*z;
		    x[n+1-i] = xm + xl*z;
        // compute weights (remember, they are symetric)
		    w[i]     = 2.0*xl/((1.0-z*z)*pp*pp);
		    w[n+1-i] = w[i];
    }
}

WITHIN_KERNEL void
gauss_legendre(ftype x1, ftype x2, int n, ftype *x, ftype *w)
{
    // Given the lower and upper limits of integration x1 and x2, and given n,
		// this routine returns arrays x[1..n] and w[1..n] of length n, containing
		// the abscissas and weights of the Gauss- Legendre n-point quadrature
		// formula.
	  // printf("n = %d\n", n);
    ftype z1,pp,p3,p2,p1;
    int m = (n+1)/2;
    ftype xm = 0.5 * (x2+x1);
    ftype xl = 0.5 * (x2-x1);

		ftype z = 0;
    for (int i=1; i<=m; i++) {
	      // printf("i = %d\n", i);
        z = cos( 3.1415926 * (i-0.25)/(n+0.5) );
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j=1; j<=n; j++) {
                p3 = p2;
                p2 = p1;
								p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
						}
            pp = n*(z*p1-p2)/(z*z-1.0);
            z1 = z;
            z  = z1-p1/pp;  // newton
        } while (fabs(z-z1) > QUADRATURE_EPS);

		    // scale to the desired interval
	      // printf("n+1-i = %d\n", n+1-i);
        x[i]     = xm - xl*z;
		    x[n+1-i] = xm + xl*z;
        // compute weights (remember, they are symetric)
		    w[i]     = 2.0*xl/((1.0-z*z)*pp*pp);
		    w[n+1-i] = w[i];
    }
}

WITHIN_KERNEL ftype
custom_function(ftype x)
{
  // return log(cos(x));
	return cos(log(x)/x)/x;
}

KERNEL void
qgaus(GLOBAL_MEM ftype *ans, ftype a, ftype b)
{
    // ftype x[6]={0.0,0.1488743389,0.4333953941, 0.6794095682,0.8650633666,0.9739065285}; 
    // ftype w[6]={0.0,0.2955242247,0.2692667193, 0.2190863625,0.1494513491,0.0666713443};
    ftype x[${nknots}]={0.}; 
    ftype w[${nknots}]={0.};
    gauss_legendre(a, b, ${nknots}, x, w);
    // The abscissas and weights. First value of each array not used.
    ftype xm = 0.5*(b+a);
		ftype xr = 0.5*(b-a);
    ftype s = 0;
    ftype dx = 0;
		ftype f_forward = 0;
		ftype f_backward = 0;
    for (int j=1; j<=5; j++) {
        dx=xr*x[j];
        // Will be twice the average value of the function, since the ten
				// weights (five numbers above each used twice) sum to 2.
        f_forward = custom_function(xm+dx);
				f_backward = custom_function(xm-dx);
        s += w[j] * ( f_forward + f_backward );
		}
		ans[0] = s * xr;
    // return s *= xr;  // scale the answer to the range of integration.
}





#endif // _MAUROANDREA_C_


// vim: fdm=marker ts=2 sw=2 sts=2 sr noet
