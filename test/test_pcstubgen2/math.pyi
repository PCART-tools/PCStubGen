"""
This module provides access to the mathematical functions
defined by the C standard.
"""
from __future__ import annotations
from _frozen_importlib import BuiltinImporter as __loader__
__all__: list[str] = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cbrt', 'ceil', 'comb', 'copysign', 'cos', 'cosh', 'degrees', 'dist', 'e', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'isqrt', 'lcm', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'nextafter', 'perm', 'pi', 'pow', 'prod', 'radians', 'remainder', 'sin', 'sinh', 'sqrt', 'sumprod', 'tan', 'tanh', 'tau', 'trunc', 'ulp']
def acos(x, /):
    """
    Return the arc cosine (measured in radians) of x.

    The result is between 0 and pi.
    """
def acosh(x, /):
    """
    Return the inverse hyperbolic cosine of x.
    """
def asin(x, /):
    """
    Return the arc sine (measured in radians) of x.

    The result is between -pi/2 and pi/2.
    """
def asinh(x, /):
    """
    Return the inverse hyperbolic sine of x.
    """
def atan(x, /):
    """
    Return the arc tangent (measured in radians) of x.

    The result is between -pi/2 and pi/2.
    """
def atan2(y, x, /):
    """
    Return the arc tangent (measured in radians) of y/x.

    Unlike atan(y/x), the signs of both x and y are considered.
    """
def atanh(x, /):
    """
    Return the inverse hyperbolic tangent of x.
    """
def cbrt(x, /):
    """
    Return the cube root of x.
    """
def ceil(x, /):
    """
    Return the ceiling of x as an Integral.

    This is the smallest integer >= x.
    """
def comb(n, k, /):
    """
    Number of ways to choose k items from n items without repetition and without order.

    Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates
    to zero when k > n.

    Also called the binomial coefficient because it is equivalent
    to the coefficient of k-th term in polynomial expansion of the
    expression (1 + x)**n.

    Raises TypeError if either of the arguments are not integers.
    Raises ValueError if either of the arguments are negative.
    """
def copysign(x, y, /):
    """
    Return a float with the magnitude (absolute value) of x but the sign of y.

    On platforms that support signed zeros, copysign(1.0, -0.0)
    returns -1.0.
    """
def cos(x, /):
    """
    Return the cosine of x (measured in radians).
    """
def cosh(x, /):
    """
    Return the hyperbolic cosine of x.
    """
def degrees(x, /):
    """
    Convert angle x from radians to degrees.
    """
def dist(p, q, /):
    """
    Return the Euclidean distance between two points p and q.

    The points should be specified as sequences (or iterables) of
    coordinates.  Both inputs must have the same dimension.

    Roughly equivalent to:
        sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
    """
def erf(x, /):
    """
    Error function at x.
    """
def erfc(x, /):
    """
    Complementary error function at x.
    """
def exp(x, /):
    """
    Return e raised to the power of x.
    """
def exp2(x, /):
    """
    Return 2 raised to the power of x.
    """
def expm1(x, /):
    """
    Return exp(x)-1.

    This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.
    """
def fabs(x, /):
    """
    Return the absolute value of the float x.
    """
def factorial(n, /):
    """
    Find n!.

    Raise a ValueError if x is negative or non-integral.
    """
def floor(x, /):
    """
    Return the floor of x as an Integral.

    This is the largest integer <= x.
    """
def fmod(x, y, /):
    """
    Return fmod(x, y), according to platform C.

    x % y may differ.
    """
def frexp(x, /):
    """
    Return the mantissa and exponent of x, as pair (m, e).

    m is a float and e is an int, such that x = m * 2.**e.
    If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.
    """
def fsum(seq, /):
    """
    Return an accurate floating-point sum of values in the iterable seq.

    Assumes IEEE-754 floating-point arithmetic.
    """
def gamma(x, /):
    """
    Gamma function at x.
    """
def gcd(*integers):
    """
    Greatest Common Divisor.
    """
def hypot(*coordinates) -> value:
    """
    Multidimensional Euclidean distance from the origin to a point.

    Roughly equivalent to:
        sqrt(sum(x**2 for x in coordinates))

    For a two dimensional point (x, y), gives the hypotenuse
    using the Pythagorean theorem:  sqrt(x*x + y*y).

    For example, the hypotenuse of a 3/4/5 right triangle is:

        >>> hypot(3.0, 4.0)
        5.0
    """
def isclose(a, b, *, rel_tol = 1e-09, abs_tol = 0.0):
    """
    Determine whether two floating-point numbers are close in value.

      rel_tol
        maximum difference for being considered "close", relative to the
        magnitude of the input values
      abs_tol
        maximum difference for being considered "close", regardless of the
        magnitude of the input values

    Return True if a is close in value to b, and False otherwise.

    For the values to be considered close, the difference between them
    must be smaller than at least one of the tolerances.

    -inf, inf and NaN behave similarly to the IEEE 754 Standard.  That
    is, NaN is not close to anything, even itself.  inf and -inf are
    only close to themselves.
    """
def isfinite(x, /):
    """
    Return True if x is neither an infinity nor a NaN, and False otherwise.
    """
def isinf(x, /):
    """
    Return True if x is a positive or negative infinity, and False otherwise.
    """
def isnan(x, /):
    """
    Return True if x is a NaN (not a number), and False otherwise.
    """
def isqrt(n, /):
    """
    Return the integer part of the square root of the input.
    """
def lcm(*integers):
    """
    Least Common Multiple.
    """
def ldexp(x, i, /):
    """
    Return x * (2**i).

    This is essentially the inverse of frexp().
    """
def lgamma(x, /):
    """
    Natural logarithm of absolute value of Gamma function at x.
    """
def log(*args, **kwargs):
    """
    Return the logarithm of x to the given base.

    If the base is not specified, returns the natural logarithm (base e) of x.
    """
def log10(x, /):
    """
    Return the base 10 logarithm of x.
    """
def log1p(x, /):
    """
    Return the natural logarithm of 1+x (base e).

    The result is computed in a way which is accurate for x near zero.
    """
def log2(x, /):
    """
    Return the base 2 logarithm of x.
    """
def modf(x, /):
    """
    Return the fractional and integer parts of x.

    Both results carry the sign of x and are floats.
    """
def nextafter(x, y, /, *, steps = None):
    """
    Return the floating-point value the given number of steps after x towards y.

    If steps is not specified or is None, it defaults to 1.

    Raises a TypeError, if x or y is not a double, or if steps is not an integer.
    Raises ValueError if steps is negative.
    """
def perm(n, k = None, /):
    """
    Number of ways to choose k items from n items without repetition and with order.

    Evaluates to n! / (n - k)! when k <= n and evaluates
    to zero when k > n.

    If k is not specified or is None, then k defaults to n
    and the function returns n!.

    Raises TypeError if either of the arguments are not integers.
    Raises ValueError if either of the arguments are negative.
    """
def pow(x, y, /):
    """
    Return x**y (x to the power of y).
    """
def prod(iterable, /, *, start = 1):
    """
    Calculate the product of all the elements in the input iterable.

    The default start value for the product is 1.

    When the iterable is empty, return the start value.  This function is
    intended specifically for use with numeric values and may reject
    non-numeric types.
    """
def radians(x, /):
    """
    Convert angle x from degrees to radians.
    """
def remainder(x, y, /):
    """
    Difference between x and the closest integer multiple of y.

    Return x - n*y where n*y is the closest integer multiple of y.
    In the case where x is exactly halfway between two multiples of
    y, the nearest even value of n is used. The result is always exact.
    """
def sin(x, /):
    """
    Return the sine of x (measured in radians).
    """
def sinh(x, /):
    """
    Return the hyperbolic sine of x.
    """
def sqrt(x, /):
    """
    Return the square root of x.
    """
def sumprod(p, q, /):
    """
    Return the sum of products of values from two iterables p and q.

    Roughly equivalent to:

        sum(itertools.starmap(operator.mul, zip(p, q, strict=True)))

    For float and mixed int/float inputs, the intermediate products
    and sums are computed with extended precision.
    """
def tan(x, /):
    """
    Return the tangent of x (measured in radians).
    """
def tanh(x, /):
    """
    Return the hyperbolic tangent of x.
    """
def trunc(x, /):
    """
    Truncates the Real x to the nearest Integral toward 0.

    Uses the __trunc__ magic method.
    """
def ulp(x, /):
    """
    Return the value of the least significant bit of the float x.
    """
e: float = 2.718281828459045
inf: float  # value = inf
nan: float  # value = nan
pi: float = 3.141592653589793
tau: float = 6.283185307179586
