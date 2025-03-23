import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
from matplotlib import cm


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


def newton_hybrid_method(func, grad_func, hess_func, x_init, eps=0.001, max_iteration=100, rho=0.6, visualize=True):
    x = x_init
    print('采用混合梯度下降法的阻尼Newton方法')
    print('待优化函数为f(x)=4x₁²+x₂²-x₁²x₂')
    print(f'初始点x0={x} 精度ε={eps}')

    # 用于保存每次迭代的点
    iterations = []
    iterations.append(x.copy())

    # 用于记录每一步使用的方法（Newton或梯度下降）
    methods = []

    for k in range(1, max_iteration):
        f = func(x)
        g = grad_func(x)
        g_norm = np.linalg.norm(g)
        H = hess_func(x)
        print(f'第{k}次迭代,x={x}')
        print(f'f(x)={f}\n▽f(x)={g} ||▽f(x)||={g_norm}\n▽²f(x)=\n{H}')
        if g_norm < eps:
            print('函数收敛程度达到精度，迭代结束')
            if visualize:
                plot_optimization_path(func, iterations, methods, x1_range=(-1, 3), x2_range=(-1, 3))
            return x
        else:
            try:  # Hd=-g
                d = np.linalg.solve(H, -g)  # 只有H可逆时才有解
                if np.dot(g, d) < 0:  # 判断是否是下降方向（H是否正定）
                    print('▽²f(x)可逆且正定，使用Newton方法下降')
                    methods.append('Newton')
                else:
                    print('▽²f(x)非正定，Newton方向非下降方向，使用梯度下降方法')
                    d = -g
                    methods.append('Grad-desc')
            except np.linalg.LinAlgError:
                print('H不可逆，使用梯度下降方法')
                d = -g
                methods.append('Grad-desc')
            l = armijo(func, grad_func, x, d, 1, rho)
            x = x + l * d
            iterations.append(x.copy())  # 保存当前迭代点

    if visualize:
        plot_optimization_path(func, iterations, methods, x1_range=(-1, 3), x2_range=(-1, 3))
    return x


def plot_optimization_path(func, iterations, methods, x1_range=(-1, 3), x2_range=(-1, 3), resolution=100):
    """
    绘制优化路径在函数图像上的可视化

    Args:
        func: 目标函数
        iterations: 优化过程中的所有点
        methods: 每一步使用的方法（Newton或梯度下降）
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
    fig = plt.figure(figsize=(15, 10))

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

    # 等高线图
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X1, X2, Z, 20, cmap=cm.viridis)
    ax2.clabel(contour, inline=True, fontsize=8)

    # 在等高线图上分段绘制优化路径，根据使用的方法区分颜色
    newton_x1, newton_x2 = [], []
    gradient_x1, gradient_x2 = [], []

    # 分类整理各段路径
    segments = []
    current_method = methods[0]
    current_segment = [(path_x1[0], path_x2[0])]

    for i in range(1, len(iterations)):
        current_segment.append((path_x1[i], path_x2[i]))
        if i == len(iterations) - 1 or methods[i - 1] != methods[i]:
            segments.append((current_method, current_segment))
            current_method = methods[i] if i < len(methods) else None
            current_segment = [(path_x1[i], path_x2[i])]

    # 绘制不同方法的路径段
    for method, segment in segments:
        x_vals = [point[0] for point in segment]
        y_vals = [point[1] for point in segment]

        if method == 'Newton':
            ax2.plot(x_vals, y_vals, 'b-o', linewidth=2, markersize=5, label='_Newton')
        else:
            ax2.plot(x_vals, y_vals, 'g-o', linewidth=2, markersize=5, label='_梯度下降')

    # 添加起点和终点标记
    ax2.plot(path_x1[0], path_x2[0], 'mo', markersize=10, label='start')
    ax2.plot(path_x1[-1], path_x2[-1], 'ro', markersize=10, label='end')

    # 创建自定义图例
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = ['start', 'end', 'damped-Newton', 'grad-desc']
    custom_handles = [
        plt.Line2D([], [], color='m', marker='o', markersize=10, linestyle=''),
        plt.Line2D([], [], color='r', marker='o', markersize=10, linestyle=''),
        plt.Line2D([], [], color='b', marker='o', markersize=5, linestyle='-'),
        plt.Line2D([], [], color='g', marker='o', markersize=5, linestyle='-')
    ]
    ax2.legend(custom_handles, unique_labels, loc='best')

    # 设置图像标题和标签
    ax2.set_title('2D visualized processing')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')

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
        method = methods[i - 1] if i > 0 else "start"
        table_data.append([i, f"{x_val[0]:.4f}", f"{x_val[1]:.4f}", f"{f_val:.4f}", method])

    # 创建表格
    table = ax3.table(cellText=table_data,
                      colLabels=['iterations', 'x₁', 'x₂', 'f(x)', 'method'],
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

    function = 4 * x1 ** 2 + x2 ** 2 - x1 * x1 * x2
    f, grad_f, hess_f = create_computational_functions(function, arguments)

    x0 = np.array([2.0, 0.0])
    x_min = newton_hybrid_method(f, grad_f, hess_f, x0)
    x_res = np.around(x_min, decimals=6)
    print('极小点x*=', x_res)
    print('极小值f(x*)=', '%.6f' % f(x_min))