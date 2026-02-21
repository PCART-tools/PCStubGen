# 存根对比报告（导入排序）

## 概览
- 旧版文件数: 1
- 新版文件数: 1
- 共同文件数: 1
- 仅旧版存在: 0
- 仅新版存在: 0

## 差异: math.pyi
```diff
--- Old:math.pyi

+++ New:math.pyi

@@ -5,53 +5,53 @@

 from __future__ import annotations
 from _frozen_importlib import BuiltinImporter as __loader__
 __all__: list[str] = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'cbrt', 'ceil', 'comb', 'copysign', 'cos', 'cosh', 'degrees', 'dist', 'e', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'isqrt', 'lcm', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'modf', 'nan', 'nextafter', 'perm', 'pi', 'pow', 'prod', 'radians', 'remainder', 'sin', 'sinh', 'sqrt', 'sumprod', 'tan', 'tanh', 'tau', 'trunc', 'ulp']
-def acos(x):
+def acos(x, /):
     """
     Return the arc cosine (measured in radians) of x.
     
     The result is between 0 and pi.
     """
-def acosh(x):
+def acosh(x, /):
     """
     Return the inverse hyperbolic cosine of x.
     """
-def asin(x):
+def asin(x, /):
     """
     Return the arc sine (measured in radians) of x.
     
     The result is between -pi/2 and pi/2.
     """
-def asinh(x):
+def asinh(x, /):
     """
     Return the inverse hyperbolic sine of x.
     """
-def atan(x):
+def atan(x, /):
     """
     Return the arc tangent (measured in radians) of x.
     
     The result is between -pi/2 and pi/2.
     """
-def atan2(y, x):
+def atan2(y, x, /):
     """
     Return the arc tangent (measured in radians) of y/x.
     
     Unlike atan(y/x), the signs of both x and y are considered.
     """
-def atanh(x):
+def atanh(x, /):
     """
     Return the inverse hyperbolic tangent of x.
     """
-def cbrt(x):
+def cbrt(x, /):
     """
     Return the cube root of x.
     """
-def ceil(x):
+def ceil(x, /):
     """
     Return the ceiling of x as an Integral.
     
     This is the smallest integer >= x.
     """
-def comb(n, k):
+def comb(n, k, /):
     """
     Number of ways to choose k items from n items without repetition and without order.
     
@@ -65,26 +65,26 @@

     Raises TypeError if either of the arguments are not integers.
     Raises ValueError if either of the arguments are negative.
     """
-def copysign(x, y):
+def copysign(x, y, /):
     """
     Return a float with the magnitude (absolute value) of x but the sign of y.
     
     On platforms that support signed zeros, copysign(1.0, -0.0)
     returns -1.0.
     """
-def cos(x):
+def cos(x, /):
     """
     Return the cosine of x (measured in radians).
     """
-def cosh(x):
+def cosh(x, /):
     """
     Return the hyperbolic cosine of x.
     """
-def degrees(x):
+def degrees(x, /):
     """
     Convert angle x from radians to degrees.
     """
-def dist(p, q):
+def dist(p, q, /):
     """
     Return the Euclidean distance between two points p and q.
     
@@ -94,64 +94,64 @@

     Roughly equivalent to:
         sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))
     """
-def erf(x):
+def erf(x, /):
     """
     Error function at x.
     """
-def erfc(x):
+def erfc(x, /):
     """
     Complementary error function at x.
     """
-def exp(x):
+def exp(x, /):
     """
     Return e raised to the power of x.
     """
-def exp2(x):
+def exp2(x, /):
     """
     Return 2 raised to the power of x.
     """
-def expm1(x):
+def expm1(x, /):
     """
     Return exp(x)-1.
     
     This function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.
     """
-def fabs(x):
+def fabs(x, /):
     """
     Return the absolute value of the float x.
     """
-def factorial(n):
+def factorial(n, /):
     """
     Find n!.
     
     Raise a ValueError if x is negative or non-integral.
     """
-def floor(x):
+def floor(x, /):
     """
     Return the floor of x as an Integral.
     
     This is the largest integer <= x.
     """
-def fmod(x, y):
+def fmod(x, y, /):
     """
     Return fmod(x, y), according to platform C.
     
     x % y may differ.
     """
-def frexp(x):
+def frexp(x, /):
     """
     Return the mantissa and exponent of x, as pair (m, e).
     
     m is a float and e is an int, such that x = m * 2.**e.
     If x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.
     """
-def fsum(seq):
+def fsum(seq, /):
     """
     Return an accurate floating-point sum of values in the iterable seq.
     
     Assumes IEEE-754 floating-point arithmetic.
     """
-def gamma(x):
+def gamma(x, /):
     """
     Gamma function at x.
     """
@@ -194,19 +194,19 @@

     is, NaN is not close to anything, even itself.  inf and -inf are
     only close to themselves.
     """
-def isfinite(x):
+def isfinite(x, /):
     """
     Return True if x is neither an infinity nor a NaN, and False otherwise.
     """
-def isinf(x):
+def isinf(x, /):
     """
     Return True if x is a positive or negative infinity, and False otherwise.
     """
-def isnan(x):
+def isnan(x, /):
     """
     Return True if x is a NaN (not a number), and False otherwise.
     """
-def isqrt(n):
+def isqrt(n, /):
     """
     Return the integer part of the square root of the input.
     """
@@ -214,13 +214,13 @@

     """
     Least Common Multiple.
     """
-def ldexp(x, i):
+def ldexp(x, i, /):
     """
     Return x * (2**i).
     
     This is essentially the inverse of frexp().
     """
-def lgamma(x):
+def lgamma(x, /):
     """
     Natural logarithm of absolute value of Gamma function at x.
     """
@@ -230,27 +230,27 @@

     
     If the base is not specified, returns the natural logarithm (base e) of x.
     """
-def log10(x):
+def log10(x, /):
     """
     Return the base 10 logarithm of x.
     """
-def log1p(x):
+def log1p(x, /):
     """
     Return the natural logarithm of 1+x (base e).
     
     The result is computed in a way which is accurate for x near zero.
     """
-def log2(x):
+def log2(x, /):
     """
     Return the base 2 logarithm of x.
     """
-def modf(x):
+def modf(x, /):
     """
     Return the fractional and integer parts of x.
     
     Both results carry the sign of x and are floats.
     """
-def nextafter(x, y, *, steps = None):
+def nextafter(x, y, /, *, steps = None):
     """
     Return the floating-point value the given number of steps after x towards y.
     
@@ -259,7 +259,7 @@

     Raises a TypeError, if x or y is not a double, or if steps is not an integer.
     Raises ValueError if steps is negative.
     """
-def perm(n, k = None):
+def perm(n, k = None, /):
     """
     Number of ways to choose k items from n items without repetition and with order.
     
@@ -272,11 +272,11 @@

     Raises TypeError if either of the arguments are not integers.
     Raises ValueError if either of the arguments are negative.
     """
-def pow(x, y):
+def pow(x, y, /):
     """
     Return x**y (x to the power of y).
     """
-def prod(iterable, *, start = 1):
+def prod(iterable, /, *, start = 1):
     """
     Calculate the product of all the elements in the input iterable.
     
@@ -286,11 +286,11 @@

     intended specifically for use with numeric values and may reject
     non-numeric types.
     """
-def radians(x):
+def radians(x, /):
     """
     Convert angle x from degrees to radians.
     """
-def remainder(x, y):
+def remainder(x, y, /):
     """
     Difference between x and the closest integer multiple of y.
     
@@ -298,19 +298,19 @@

     In the case where x is exactly halfway between two multiples of
     y, the nearest even value of n is used. The result is always exact.
     """
-def sin(x):
+def sin(x, /):
     """
     Return the sine of x (measured in radians).
     """
-def sinh(x):
+def sinh(x, /):
     """
     Return the hyperbolic sine of x.
     """
-def sqrt(x):
... 已截断
```

