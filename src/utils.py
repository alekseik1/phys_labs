import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

def plot_data_optimized(x_values, y_values, fig=None, axes=None,
                        title=r'Graph', x_label=r'X', y_label=r'Y', 
                        save_to=False, threshold=0.0):
    '''
    Умное построение:
    1) Убираются данные, похожие на шум (те, чье значение < 1% от максимального)
    '''
    if fig is None or axes is None:
        fig, axes = plt.subplots(figsize=(16, 9))
    
    max_y, min_y = np.max(y_values), np.min(y_values)
    threshold *= (max_y-min_y)
    # Перед тем, как менять данные, сделаем их копию
    x_values, y_values = x_values.copy(), y_values.copy()
    x_values, y_values = (x_values[y_values >= threshold], 
                          y_values[y_values >= threshold])
    
    axes.plot(x_values, y_values)
    axes.grid(True)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    
    # Сохраним
    if save_to:
        fig.savefig(save_to)
    return fig, axes

from scipy.optimize import curve_fit
def fit_with_gauss(x_data, y_data, y_error=None, initial_guess=[1, 1, 1]):
    #def find_optimal_function(data, functions, y_data='N', x_data='LL', y_error=None):
    if y_error is None:
        #y_error = y_data + '_error'
        y_error = np.zeros(len(y_data))
    #  Подготим все к подгону! 1) Вычтем минимальное значение;
    # 2) поделим X на максимальный из X;
    y_data = y_data.copy()
    x_data = x_data.copy()
    min_y = np.min(y_data)
    y_data -= min_y
    max_x = np.max(x_data)
    x_data /= max_x
    func = lambda x, a, mu, sigma: a*np.exp(-(x-mu)**2/(2*sigma**2))
    popt, pcov = curve_fit(func, x_data, y_data, maxfev=100000, p0=initial_guess)
    # Преобразуем оптимальные параметры обратно
    popt[1] *= max_x
    popt[2] *= max_x
    return lambda x: (func(x , *popt) + min_y), popt
