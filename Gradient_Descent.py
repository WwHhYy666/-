import sympy as sp
import numpy as np

# 定义符号变量和表达式
x1, x2 = sp.symbols('x1 x2')
symbols = sp.Array([x1, x2])
expr = sp.sin(x1 * x2) + sp.cos(x1) + sp.tan(x2)


def create_function(function, variables):
    """
    生成可代入数值计算函数值的函数
    输入:
        function: sympy 表达式
        variables: sympy 符号向量 (Array格式)
    输出:
        接收 NumPy 数组输入并返回计算结果的函数
    """
    # 将 sympy 表达式转换为可调用的 lambda 函数
    func = sp.lambdify(variables, function, 'numpy')
    # 包装为接收 NumPy 数组的接口
    return lambda x0: func(*x0)


def create_gradient(function, variables):
    """
    生成可代入数值计算梯度的函数
    输入:
        function: sympy 表达式
        variables: sympy 符号向量 (Array格式)
    输出:
        接收 NumPy 数组输入并返回梯度向量的函数
    """
    gradient = [sp.diff(function, s) for s in variables]
    grad = sp.lambdify(variables, gradient, 'numpy')

    def _gradient_wrapper(x0):
        """梯度计算包装函数"""
        # 计算原始梯度
        grad_func = grad(*x0)
        grad_array = np.array(grad_func, dtype=np.float64).flatten()
        return grad_array

    return _gradient_wrapper


def armijo(x, d, alpha, rho, k):
    print('     第', k, '次迭代:')
    if k >= 500:
        print('达到迭代阈值，终止迭代')
        return alpha
    else:
        print('     步长α=', alpha)
        f1 = f(x + alpha * d)
        f2 = f(x) + rho * alpha * np.dot(grad_f(x), d)
        print('     f(x+αd)=', f1)
        print('     f(x)+ρα▽f(x)Td=', f2)
        if f1 <= f2:
            print('     符合Armijo停止条件，得最终步长α=', alpha)
            return alpha
        else:
            print('     不符合Armijo停止条件，继续迭代')
            return armijo(x, d, alpha * rho, rho, k+1)


def grad_desc(x, rho, eps, k):
    print('第', k, '次迭代:')
    if k >= 900:
        print('达到迭代阈值，终止迭代')
        print('找到极小点x=', x)
        print('极小值f(x)=', f(x))
        return
    else:
        grad = grad_f(x)
        d = -grad
        print('位于', x, 'f(x)=', f(x), '▽f(x)=', grad)
        print('选择下降方向d=', d)
        print('使用Armijo策略求步长(ρ=0.6):')
        alpha = armijo(x, d, 1, rho, 1)
        f1 = f(x + alpha * d)
        f2 = f(x) + rho * alpha * np.dot(grad_f(x), d)
        if abs(f1 - f2) <= eps:
            print('达到所需精度，停止迭代')
            print('找到极小点x=', x)
            print('极小值f(x)=', f(x))
            return
        else:
            print('未达到要求精度，继续迭代')
            grad_desc(x + alpha * d, rho, eps, k + 1)


if __name__ == '__main__':
    # 生成函数和梯度计算函数
    f = create_function(expr, symbols)
    grad_f = create_gradient(expr, symbols)

    # 输入数值并计算结果
    point = np.array([1.0, -1.0])  # 注意输入类型为浮点数
    print('使用梯度下降法求极小点(ε=1e-15)')
    grad_desc(point, 0.6, 1e-15, 1)
