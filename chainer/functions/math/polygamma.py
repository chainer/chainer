from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check


_polygamma_cpu = None
_digamma_kernel = None
_zeta_kernel = None
_gamma_kernel = None


zeta_definition = '''
/* Expansion coefficients
 * for Euler-Maclaurin summation formula
 * (2k)! / B2k
 * where B2k are Bernoulli numbers
 */
static __device__ double A[] = {
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9,	/*1.307674368e12/691 */
    7.47242496e10,
    -2.950130727918164224e12,	/*1.067062284288e16/3617 */
    1.1646782814350067249e14,	/*5.109094217170944e18/43867 */
    -4.5979787224074726105e15,	/*8.028576626982912e20/174611 */
    1.8152105401943546773e17,	/*1.5511210043330985984e23/854513 */
    -7.1661652561756670113e18	/*1.6938241367317436694528e27/236364091 */
};

static __device__ double MACHEP = 1.11022302462515654042E-16;

/* 30 Nov 86 -- error in third coefficient fixed */


double __device__ zeta(double x, double q)
{
    int i;
    double a, b, k, s, t, w;

    if (x == 1.0)
    goto retinf;

    if (x < 1.0) {
      domerr:
    return nan("");
    }

    if (q <= 0.0) {
    if (q == floor(q)) {
      retinf:
        return 1.0/0.0;
    }
    if (x != floor(x))
        goto domerr;	/* because q^-x not defined */
    }

    /* Asymptotic expansion
     * http://dlmf.nist.gov/25.11#E43
     */
    if (q > 1e8) {
        return (1/(x - 1) + 1/(2*q)) * pow(q, 1 - x);
    }

    /* Euler-Maclaurin summation formula */

    /* Permit negative q but continue sum until n+q > +9 .
     * This case should be handled by a reflection formula.
     * If q<0 and x is an integer, there is a relation to
     * the polyGamma function.
     */
    s = pow(q, -x);
    a = q;
    i = 0;
    b = 0.0;
    while ((i < 9) || (a <= 9.0)) {
        i += 1;
        a += 1.0;
        b = pow(a, -x);
        s += b;
        if (fabs(b / s) < MACHEP)
            goto done;
    }

    w = a;
    s += b * w / (x - 1.0);
    s -= 0.5 * b;
    a = 1.0;
    k = 0.0;
    for (i = 0; i < 12; i++) {
        a *= x + k;
        b /= w;
        t = a * b / A[i];
        s = s + t;
        t = fabs(t / s);
        if (t < MACHEP)
            goto done;
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
done:
    return (s);
}
'''


polevl_definition = '''
static __device__ double polevl(double x, double coef[], int N)
{
    double ans;
    int i;
    double *p;

    p = coef;
    ans = *p++;
    i = N;

    do
    ans = ans * x + *p++;
    while (--i);

    return (ans);
}
'''


psi_definition = '''
static __device__ double A[] = {
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2
};

static __device__ double PI = 3.141592653589793;
static __device__ double EULER = 0.5772156649015329;

static __device__ double digamma_imp_1_2(double x)
{
    /*
     * Rational approximation on [1, 2] taken from Boost.
     *
     * Now for the approximation, we use the form:
     *
     * digamma(x) = (x - root) * (Y + R(x-1))
     *
     * Where root is the location of the positive root of digamma,
     * Y is a constant, and R is optimised for low absolute error
     * compared to Y.
     *
     * Maximum Deviation Found:               1.466e-18
     * At double precision, max error found:  2.452e-17
     */
    double r, g;

    static const float Y = 0.99558162689208984f;

    static const double root1 = 1569415565.0 / 1073741824.0;
    static const double root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
    static const double root3 = 0.9016312093258695918615325266959189453125e-19;

   static double P[] = {
       -0.0020713321167745952,
       -0.045251321448739056,
       -0.28919126444774784,
       -0.65031853770896507,
       -0.32555031186804491,
       0.25479851061131551
   };
   static double Q[] = {
       -0.55789841321675513e-6,
       0.0021284987017821144,
       0.054151797245674225,
       0.43593529692665969,
       1.4606242909763515,
       2.0767117023730469,
       1.0
   };
   g = x - root1;
   g -= root2;
   g -= root3;
   r = polevl(x - 1.0, P, 5) / polevl(x - 1.0, Q, 6);

   return g * Y + g * r;
}


static __device__ double psi_asy(double x)
{
    double y, z;

    if (x < 1.0e17) {
    z = 1.0 / (x * x);
    y = z * polevl(z, A, 6);
    }
    else {
    y = 0.0;
    }

    return log(x) - (0.5 / x) - y;
}


double __device__ psi(double x)
{
    double y = 0.0;
    double q, r;
    int i, n;

    if (isnan(x)) {
    return x;
    }
    else if (isinf(x)){
        if(x > 0){
            return x;
        }else{
            return nan("");
        }
    }
    else if (x == 0) {
        return -1.0/0.0;
    }
    else if (x < 0.0) {
    /* argument reduction before evaluating tan(pi * x) */
    r = modf(x, &q);
    if (r == 0.0) {
        return nan("");
    }
    y = -PI / tan(PI * r);
    x = 1.0 - x;
    }

    /* check for positive integer up to 10 */
    if ((x <= 10.0) && (x == floor(x))) {
    n = (int)x;
    for (i = 1; i < n; i++) {
        y += 1.0 / i;
    }
    y -= EULER;
    return y;
    }

    /* use the recurrence relation to move x into [1, 2] */
    if (x < 1.0) {
    y -= 1.0 / x;
    x += 1.0;
    }
    else if (x < 10.0) {
    while (x > 2.0) {
        x -= 1.0;
        y += 1.0 / x;
    }
    }
    if ((1.0 <= x) && (x <= 2.0)) {
    y += digamma_imp_1_2(x);
    return y;
    }

    /* x is large, use the asymptotic series */
    y += psi_asy(x);
    return y;
}
'''


class PolyGamma(function_node.FunctionNode):

    @property
    def label(self):
        return 'polygamma'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        n_type, x_type = in_types

        type_check.expect(
            n_type.dtype.kind == 'i',
            x_type.dtype.kind == 'f',
        )

    def forward_cpu(self, inputs):
        n, x = inputs
        global _polygamma_cpu
        if _polygamma_cpu is None:
            try:
                from scipy import special
                _polygamma_cpu = special.polygamma
            except ImportError:
                raise ImportError("SciPy is not available. Forward computation"
                                  " of polygamma can not be done.")
        self.retain_inputs((0, 1))
        return utils.force_array(_polygamma_cpu(n, x), dtype=x.dtype),

    def forward_gpu(self, inputs):
        global _digamma_kernel
        global _gamma_kernel
        global _zeta_kernel
        n, x = inputs
        self.retain_inputs((0, 1))
        if _digamma_kernel is None:
            _digamma_kernel = cuda.elementwise(
                'T x', 'T y',
                'y = psi(x)',
                'elementwise_digamma',
                preamble=polevl_definition+psi_definition
            )
        if _gamma_kernel is None:
            _gamma_kernel = cuda.elementwise(
                'T x', 'T y',
                'y = tgamma(x)',
                'elementwise_gamma'
            )
        if _zeta_kernel is None:
            _zeta_kernel = cuda.elementwise(
                'T x, T q', 'T y',
                'y = zeta(x, q)',
                'elementwise_zeta',
                preamble=zeta_definition
            )
        n, x = cuda.cupy.broadcast_arrays(n, x)
        n_xtype = n.astype(x.dtype)
        fac2 = (-1.0)**(n+1).astype(x.dtype) * _gamma_kernel(n_xtype+1.0) \
            * _zeta_kernel(n_xtype+1.0, x)
        return cuda.cupy.where(n == 0, _digamma_kernel(x), fac2),

    def backward(self, indexes, gy):
        n, x = self.get_retained_inputs()
        return None, polygamma(n + 1, x) * gy[0],


def polygamma(n, x):
    """Polygamma function.

    .. note::
       Forward computation can not be done if
       `SciPy <https://www.scipy.org/>`_ is not available.

    Args:
        n (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return PolyGamma().apply((n, x))[0]
