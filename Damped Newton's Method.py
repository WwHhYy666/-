import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify


def create_computational_functions(expr, variables):
    grad_expr = [sp.diff(expr, var) for var in variables]
    hess_expr = [[sp.diff(g, var) for var in variables] for g in grad_expr]

    func = lambdify(variables, expr, "numpy")
    grad = lambdify(variables, grad_expr, "numpy")
    hess = lambdify(variables, hess_expr, "numpy")

    def func_wrapper(x):
        return func(*x)

    def grad_wrapper(x):
        return np.array(grad(*x))

    def hess_wrapper(x):
        return np.array(hess(*x))

    return func_wrapper, grad_wrapper, hess_wrapper


def armijo(func, grad_func, x, d, alpha_init, rho, max_iter=100):
    alpha = alpha_init
    fx = func(x)
    grad_fx = grad_func(x)
    print(f'     采用Armijo线搜索准则求步长，ρ={rho}')
    print(f'     x={x},d={d}')
    for k in range(1, max_iter):
        if k == max_iter:
            print('     达到最大迭代次数，终止迭代')
            break
        print(f'     第{k}次迭代：')
        print(f'     步长α={alpha}')
        fx_next = func(x + alpha * d)
        fx_jg = fx + rho * alpha * np.dot(grad_fx, d)
        print(f'     f(x+αd)={fx_next}')
        print(f'     f(x)+ραgᵀ(x)d={fx_jg}')
        if fx_next <= fx_jg:
            print('     达到Armijo停止条件，终止迭代')
            break
        print('     α不满足停止条件，继续迭代')
        alpha = alpha * rho
    return alpha


def damped_newton_method(func, grad_func, hess_func, x0, eps=0.001, max_iteration=100, rho=0.6):
    x = x0
    print('采用Armijo线搜索的阻尼Newton方法')
    print('待优化函数为f(x)=4x₁²+x₂²-8x₁-4x₂')
    print(f'初始点x0={x0}精度ε={eps}')
    for k in range(1, max_iteration):
        f = func(x)
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        H = hess_func(x)
        print(f'第{k}次迭代')
        print(f'x={x}')
        print(f'f(x)={f}\n▽f(x)={g}||▽f(x)||={g_norm}\n▽²f(x)=\n{H}')
        if np.linalg.norm(g) < eps:
            print('函数收敛程度达到精度，迭代结束')
            return x
        else:
            d = np.linalg.solve(H, -g)
            l = armijo(func, grad_func, x, d, 1, rho)
            x = x + l * d


if __name__ == "__main__":
    x1, x2 = sp.symbols('x1 x2')
    arguments = [x1, x2]

    function = 4 * x1 ** 2 + x2 ** 2 - 8 * x1 - 4 * x2
    f, grad_f, hess_f = create_computational_functions(function, arguments)

    x_init = np.array([0, 0])
    x_min = damped_newton_method(f, grad_f, hess_f, x_init)
    x_res = np.around(x_min, decimals=6)
    print(f'求得极小点x={x_res}')
    print('极小值f(x)=', '%.6f' % f(x_min))
