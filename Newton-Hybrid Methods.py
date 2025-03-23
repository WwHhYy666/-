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


def newton_hybrid_method(func, grad_func, hess_func, x_init, eps=0.001, max_iteration=100, rho=0.6):
    x = x_init
    print('采用混合梯度下降法的阻尼Newton方法')
    print('待优化函数为f(x)=4x₁²+x₂²-x₁²x₂')
    print(f'初始点x0={x} 精度ε={eps}')
    for k in range(1, max_iteration):
        f = func(x)
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        H = hess_func(x)
        print(f'第{k}次迭代,x={x}')
        print(f'f(x)={f}\n▽f(x)={g} ||▽f(x)||={g_norm}\n▽²f(x)=\n{H}')
        if g_norm < eps:
            print('函数收敛程度达到精度，迭代结束')
            return x
        else:
            try:  # Hd=-g
                d = np.linalg.solve(H, -g)  # 只有H可逆时才有解
                if np.dot(g, d) < 0:  # 判断是否是下降方向（H是否正定）
                    print('▽²f(x)可逆且正定，使用Newton方法下降')
                else:
                    print('▽²f(x)非正定，Newton方向非下降方向，使用梯度下降方法')
                    d = -g
            except np.linalg.LinAlgError:
                print('H不可逆，使用梯度下降方法')
                d = -g
            l = armijo(func, grad_func, x, d, 1, rho)
            x = x + l * d


if __name__ == "__main__":
    x1, x2 = sp.symbols('x1 x2')
    arguments = [x1, x2]

    function = 4 * x1 ** 2 + x2 ** 2 - x1 * x1 * x2
    f, grad_f, hess_f = create_computational_functions(function, arguments)

    x0 = np.array([2.0, 0.0])
    x_min = newton_hybrid_method(f, grad_f, hess_f, x0)
    x_res = np.around(x_min, decimals=6)
    print('极小点x*=', x_res)
    print('极小值f(x*)=', '%.6f' % f(x_min))
