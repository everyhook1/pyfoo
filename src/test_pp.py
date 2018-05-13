from sympy.functions import sin, cos,exp
from sympy import init_printing, symbols, Function
init_printing()

x, h = symbols("x,h")
f = exp(x)

pprint(f(x).series(x, x0=h, n=3))
