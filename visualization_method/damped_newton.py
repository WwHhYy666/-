import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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


def damped_newton_method(func, grad_func, hess_func, x0, eps=0.001, max_iteration=100, rho=0.6, visualize=True):
    x = x0
    print('采用Armijo线搜索的阻尼Newton方法')
    print('待优化函数为f(x)=4x₁²+x₂²-8x₁-4x₂')
    print(f'初始点x0={x0}精度ε={eps}')

    # 用于保存每次迭代的点
    iterations = []
    iterations.append(x.copy())

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
            if visualize:
                plot_optimization_path(func, iterations, x1_range=(-1, 3), x2_range=(-1, 3))
            return x
        else:
            d = np.linalg.solve(H, -g)
            l = armijo(func, grad_func, x, d, 1, rho)
            x = x + l * d
            iterations.append(x.copy())  # 保存当前迭代点

    if visualize:
        plot_optimization_path(func, iterations, x1_range=(-1, 3), x2_range=(-1, 3))
    return x


def plot_optimization_path(func, iterations, x1_range=(-1, 3), x2_range=(-1, 3), resolution=100):
    """
    绘制优化路径在函数图像上的可视化
    Args:
        func: 目标函数
        iterations: 优化过程中的所有点
        x1_range: x1坐标的显示范围
        x2_range: x2坐标的显示范围
        resolution: 网格分辨率
    """
    # 创建网格
    x1 = np.linspace(x1_range[0], x1_range[1], resolution)
    x2 = np.linspace(x2_range[0], x2_range[1], resolution)
    X1, X2 = np.meshgrid(x1, x2)

    # 计算函数值
    Z = np.zeros_like(X1)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X1[i, j], X2[i, j]]))

    # 创建图形
    fig = plt.figure(figsize=(12, 10))

    # 3D曲面图
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)

    # 提取迭代点的坐标
    path_x1 = [point[0] for point in iterations]
    path_x2 = [point[1] for point in iterations]
    path_z = [func(point) for point in iterations]

    # 绘制优化路径
    ax1.plot(path_x1, path_x2, path_z, 'r-o', linewidth=2, markersize=5)

    # 设置图像标题和标签
    ax1.set_title('3D visualized processing')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$f(x_1, x_2)$')

    # 2D等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X1, X2, Z, 20, cmap=cm.viridis)
    ax2.clabel(contour, inline=True, fontsize=8)

    # 在等高线图上绘制优化路径
    ax2.plot(path_x1, path_x2, 'r-o', linewidth=2, markersize=5)

    # 添加起点和终点标记
    ax2.plot(path_x1[0], path_x2[0], 'go', markersize=10, label='start')
    ax2.plot(path_x1[-1], path_x2[-1], 'bo', markersize=10, label='end')

    # 设置图像标题和标签
    ax2.set_title('2D visualized processing')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.legend()

    plt.colorbar(contour, ax=ax2, label='f(x)')
    plt.tight_layout()

    # 添加一个额外子图显示迭代数据表格
    plt.figure(figsize=(12, 6))
    ax3 = plt.subplot(111)
    ax3.axis('tight')
    ax3.axis('off')

    # 准备表格数据
    table_data = []
    for i in range(len(iterations)):
        x_val = iterations[i]
        f_val = func(x_val)
        table_data.append([i, f"{x_val[0]:.4f}", f"{x_val[1]:.4f}", f"{f_val:.4f}"])

    # 创建表格
    table = ax3.table(cellText=table_data,
                      colLabels=['iterations', 'x₁', 'x₂', 'f(x)'],
                      loc='center',
                      cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.title('optimization processing')
    plt.tight_layout()

    plt.show(block=True)


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
