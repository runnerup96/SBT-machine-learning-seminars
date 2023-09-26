import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
import numpy as np
from itertools import zip_longest

class TrajectoryAnimation(animation.FuncAnimation):
    """
    Класс для анимации траекторий.

    Параметры:
    paths (tuple of array-like): Кортеж с траекториями для анимации.
    labels (list, optional): Список меток для линий на графике (по умолчанию пустой).
    skip_samples (int, optional): Количество пропускаемых образцов между кадрами анимации (по умолчанию 500).
    fig (matplotlib.figure.Figure, optional): Объект графика Matplotlib (по умолчанию None).
    ax (matplotlib.axes.Axes, optional): Объект осей Matplotlib (по умолчанию None).
    frames (int, optional): Количество кадров в анимации (по умолчанию наибольшее число образцов среди всех траекторий).
    interval (int, optional): Интервал между кадрами анимации в миллисекундах (по умолчанию 60).
    repeat_delay (int, optional): Задержка перед повторением анимации в миллисекундах (по умолчанию 5).
    blit (bool, optional): Флаг, указывающий, следует ли использовать "blit" для улучшения производительности (по умолчанию True).

    Атрибуты:
    fig (matplotlib.figure.Figure): Объект графика Matplotlib.
    ax (matplotlib.axes.Axes): Объект осей Matplotlib.
    paths (tuple of array-like): Кортеж с траекториями для анимации.
    lines (list of matplotlib.lines.Line2D): Список линий для отображения траекторий.
    points (list of matplotlib.lines.Line2D): Список точек для отображения текущей позиции на траекториях.

    Методы:
    init_anim(): Инициализация начальных значений линий и точек.
    animate(i): Функция для анимации на i-м кадре.

    Пример использования:
    ```
    anim = TrajectoryAnimation(path1, path2, labels=['Path 1', 'Path 2'])
    plt.show()
    ```

    """
    
    def __init__(self, *paths, labels=[], skip_samples=500, fig=None, ax=None, frames=None, 
                 interval=60, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        
        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
  
        self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                      for _, label in zip_longest(paths, labels)]
        self.points = [ax.plot([], [], '--o', color=line.get_color())[0] 
                       for line in self.lines]

        self.skip_samples = skip_samples

        super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        return self.lines + self.points

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[::,:i*self.skip_samples])
            point.set_data(*path[::,(i-1)*self.skip_samples:i*self.skip_samples])
        return self.lines + self.points
    
    
def prepare_2d_countor_plot(x, y, z, x_lim, y_lim, minimum):
    """
    Подготавливает и возвращает график с контурами для двухмерных данных.

    Параметры:
    x (array-like): Массив значений по оси X.
    y (array-like): Массив значений по оси Y.
    z (array-like): Массив значений функции Z(x, y).
    x_lim (tuple): Границы по оси X (минимум и максимум).
    y_lim (tuple): Границы по оси Y (минимум и максимум).
    minimum (array-like): Координаты минимума функции (x, y).

    Возвращает:
    ax: Объект графика Matplotlib.
    """
    minima = minimum.reshape(-1, 1)
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
    ax.plot(*minima, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return ax
    

def prepare_beale_3d_countor_plot(x, y, z, f):
    """
    Подготавливает и возвращает трехмерный график с контурами для функции Beale.

    Параметры:
    x (array-like): Массив значений по оси X.
    y (array-like): Массив значений по оси Y.
    z (array-like): Массив значений функции Z(x, y).
    f (function): Функция Beale.

    Возвращает:
    ax: Объект трехмерного графика Matplotlib.
    """
    minima = np.array([3., .5]).reshape(-1, 1)
    xmin, xmax = -4.5, 4.5
    ymin, ymax = -4.5, 4.5
    
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d', elev=30, azim=-50)

    ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, 
                    edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    ax.plot(*minima, f(*minima), 'r*', markersize=10)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return ax
