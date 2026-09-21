# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from IPython.display import Math, display
from sympy import lambdify
from scipy.signal import find_peaks

def set_preferences():
    """
    Set the preferences for matplotlib plots.
    This function sets various preferences for matplotlib plots, such as font family, figure size, label size, grid lines, and legend style.
    Returns:
        None
    """    
    rc('font',**{'family':'serif','serif':['Times']})
    rc('text', usetex=True)

    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (4,2)

    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['font.size'] = 10

    plt.rcParams['axes.linewidth'] =  0.5
    plt.rcParams['grid.linewidth'] =  0.5
    plt.rcParams['lines.linewidth'] =  0.5
    plt.rcParams['lines.markersize'] =  2
                    
    # Grid lines
    plt.rcParams['axes.grid'] =   False
    plt.rcParams['axes.axisbelow'] =  False
    plt.rcParams['grid.linestyle'] =  'dashed'
    plt.rcParams['grid.color'] =   'k'
    plt.rcParams['grid.alpha'] =   0.25
    plt.rcParams['grid.linewidth'] =   0.5

    # Legend
    plt.rcParams['legend.frameon'] =   False
    plt.rcParams['legend.framealpha'] =   0.25
    plt.rcParams['legend.fancybox'] =   False
    plt.rcParams['legend.numpoints'] =   1

    return None

def symdisp(expr, var, unit=" "):
    """
    Display sympy expressions in Latex style.

    :param expr: expression in latex [string]
    :param var: sympy variable, function, expression.
    :param unit: string indicating unit of var [string]
    """
    display(Math(expr + sp.latex(var) + "\;" + "\mathrm{"+unit+"}"))


# função para arredondamento de floats em expressões simbólicas
def round_expr(expr, numDig):
    """
    Rounds numerical values in sympy expressions

    :param expr: sympy symbolic expression
    :param numDig: number of rounding decimals

    :return: rounded expression
    """
    return expr.xreplace({n: round(n, numDig) for n in expr.atoms(sp.Number)})


# Função para plot de funções do sympy
def symplot(t, F, interval, funLabel, xlabel=" tempo [s]", ylabel="", figsize=None, xfactor=None, yfactor=None, returnAxis=False):
    """
    Plot sympy expressions.

    Parameters
    ----------
    t : sympy.core.symbol.Symbol
        Time variable.
    F : sympy.core.basic.Basic
        Sympy expression.
    interval : np.array
        Time interval.
    funLabel : str
        Label for the function.
    xlabel : str, optional
        Label for the x-axis. Default is " tempo [s]".
    ylabel : str, optional
        Label for the y-axis. Default is "".
    figsize : tuple, optional
        Figure size. Default is None.
    xfactor : float, optional
        Factor to scale the x-axis. Default is None.
    yfactor : float, optional
        Factor to scale the y-axis. Default is None.

    Returns 
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    if xfactor is None:
        xfactor = 1

    if yfactor is None:
        yfactor = 1

    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    if type(F) == list:
        if type(yfactor) == list:
            for indLabel, f in enumerate(F):
                plotFunc(t, f, interval, funLabel[indLabel], xlabel, ylabel, xfactor, yfactor[indLabel])
        else:
            for indLabel, f in enumerate(F):
                plotFunc(t, f, interval, funLabel[indLabel], xlabel, ylabel, xfactor, yfactor)
    else:
        plotFunc(t, F, interval, funLabel, xlabel, ylabel, xfactor, yfactor)

    ax = plt.gca()

    plt.grid()
    #plt.close()
    
    if returnAxis:
        return fig, ax
    else:
        return fig


def plotFunc(t, F, interval, funLabel, xlabel, ylabel, xfactor, yfactor):
    """
    Plots a given function F over a specified interval.

    Parameters
    ----------

    t : sympy.Symbol
        The symbolic variable used in the function F.
    F : sympy.Expr
        The symbolic expression representing the function to be plotted.
    interval : numpy.ndarray
        The range of values over which to evaluate and plot the function.
    funLabel : str
        The label for the function to be used in the plot legend.
    xlabel : str
        The label for the x-axis of the plot.
    ylabel : str
        The label for the y-axis of the plot.
    xfactor : float
        The factor by which to scale the x-axis values.
    yfactor : float
        The factor by which to scale the y-axis values.
    
    Returns
    -------
    None

    """    
    func = lambdify(
        t, F, modules=["numpy", {"Heaviside": lambda t: np.heaviside(t, 0)}]
    )
    f_num = func(interval)

    # make sure discontinuities are not plotted
    f_diff = np.abs(np.diff(f_num))   
    f_diff = np.concatenate((f_diff, [0]))
    peaks, _ = find_peaks(f_diff, width=[0,2],height=0) 
    f_num[peaks] = np.nan
   
    plt.plot(interval/xfactor, f_num/yfactor, label=funLabel)
    plt.legend()
    plt.xlim([min(interval/xfactor), max(interval/xfactor)])
    plt.xlabel(f'${xlabel}$')
    plt.ylabel(ylabel)

def genGIF(x, y, figName, xlabel=[], ylabel=[], fram=200, inter=20):
    """
    Create and save a plot animation as GIF

    :param x: x-axis values [np array]
    :param y: y-axis values [np array]
    :param figName: figure file name w/ folder path [string]
    :param xlabel: xlabel [string]
    :param ylabel: ylabel [string]
    :param fram: number of frames [int]
    :param inter: time interval between frames [milliseconds]

    """
    figAnin = plt.figure()
    ax = plt.axes(
        xlim=(np.min(x), np.max(x)),
        ylim=(
            np.min(y) - 0.1 * np.max(np.abs(y)),
            np.max(y) + 0.1 * np.max(np.abs(y)),
        ),
    )
    (line,) = ax.plot([], [])
    ax.grid()

    indx = np.arange(0, len(x), int(len(x) / fram))

    if len(xlabel):
        plt.xlabel(xlabel)

    if len(ylabel):
        plt.ylabel(ylabel)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(x[:indx[i]], y[:indx[i]])
        return (line,)

    anim = FuncAnimation(
        figAnin,
        animate,
        init_func=init,
        frames=fram,
        interval=inter,
        blit=True,
    )

    anim.save(figName, dpi=200, writer="imagemagick")
    plt.close()

def genConvGIF(
    x,
    h,
    t,
    totalTime,
    ti,
    tf,
    figName,
    xlabel=[],
    ylabel=[],
    fram=200,
    inter=20,
    plotConv=False,
):
    """
    Create and save a convolution plot animation as GIF

    :param x: x(t) function [sympy expr]
    :param h: h(t) function [sympy expr]
    :param t: t time variable [sympy variable]
    :param totalTime: array of time instants where the functions will be evaluated [nparray]
    :param ti: time when animation starts [scalar]
    :param tf: time when animation stops [scalar]
    :param figName: figure file name w/ folder path [string]
    :param xlabel: xlabel [string]
    :param ylabel: ylabel [string]
    :param fram: number of frames [int]
    :param inter: time interval between frames [milliseconds]

    """
    x_func = lambdify(
        t, x, modules=["numpy", {"Heaviside": lambda t: np.heaviside(t, 0)}]
    )
    h_func = lambdify(
        t, h, modules=["numpy", {"Heaviside": lambda t: np.heaviside(t, 0)}]
    )

    x_num = x_func(totalTime)
    h_num = h_func(totalTime)
    dt = totalTime[1] - totalTime[0]

    if plotConv:
        y_num = np.convolve(h_num, x_num, "same") * dt
        ymax = np.max([x_num, h_num, y_num])
        ymin = np.min([x_num, h_num, y_num])
    else:
        ymax = np.max([x_num, h_num])
        ymin = np.min([x_num, h_num])

    figAnim = plt.figure()
    ax = plt.axes(
        xlim=(totalTime.min(), totalTime.max()),
        ylim=(ymin - 0.1 * np.abs(ymax), ymax + 0.1 * np.abs(ymax)),
    )
    line1, line2, line3 = ax.plot([], [], [], [], [], [])
    line1.set_label(ylabel[0])
    line2.set_label(ylabel[1])

    if plotConv:
        line3.set_label(ylabel[2])

    ax.grid()
    ax.legend(loc="upper right")

    # plot static function
    figh = symplot(t, h, totalTime, "h(t)")

    if len(xlabel):
        plt.xlabel(xlabel)

    def init():
        line1.set_data(figh.get_axes()[0].lines[0].get_data())
        return (line1,)

    plt.close(figh)

    delays = totalTime[:: int(len(totalTime) / fram)]
    ind = np.arange(0, len(totalTime), int(len(totalTime) / fram))

    ind = ind[delays > ti]
    delays = delays[delays > ti]

    ind = ind[delays < tf]
    delays = delays[delays < tf]

    totalFrames = len(delays)

    def animate(i):
        figx = symplot(t, x.subs({t: delays[i] - t}), totalTime, "x(t-τ)")
        line2.set_data(figx.get_axes()[0].lines[0].get_data())

        if plotConv:
            line3.set_data(totalTime[:ind[i]], y_num[:ind[i]])

        plt.close(figx)
        return line2, line3

    anim = FuncAnimation(
        figAnim,
        animate,
        init_func=init,
        frames=totalFrames,
        interval=inter,
        blit=True,
    )

    anim.save(figName, dpi=200, writer="imagemagick")
    plt.close()

def plot_modulacoes_digitais(symbols, M=2, carrier_cycles_per_symbol=4):
    """
    Gera os gráficos das modulações digitais (ASK, FSK, PSK) para qualquer ordem M.
    
    Parâmetros:
    -----------
    symbols : list ou np.array
        Sequência de símbolos a serem transmitidos (valores entre 0 e M - 1).
    M : int
        Ordem da modulação digital (ex.: 2 para binário, 4 para 4-ário, 8, etc.).
    carrier_cycles_per_symbol : int
        Número de ciclos da portadora por período de símbolo.
    """
    symbols = np.array(symbols, dtype=int)
    n_symbols = len(symbols)
    samples_per_symbol = 500
    total_samples = n_symbols * samples_per_symbol
    
    t = np.linspace(0, n_symbols, total_samples, endpoint=False)
    
    # -------------------------------------------------------------
    # 1. Sinal Digital em Banda Básica
    # -------------------------------------------------------------
    # Mapeamento de níveis para visualização dos pulsos (-1 a +1)
    if M == 2:
        levels = np.where(symbols == 1, 1.0, -1.0)
    else:
        levels = -1.0 + 2.0 * symbols / (M - 1)
        
    t_digital = []
    y_digital = []
    for i, val in enumerate(levels):
        t_digital.extend([i, i + 1])
        y_digital.extend([val, val])
        
    # Fechamento inicial e final no nível zero
    t_digital = [0] + t_digital + [n_symbols]
    y_digital = [0] + y_digital + [0]
    
    # -------------------------------------------------------------
    # 2. Portadora de Alta Frequência
    # -------------------------------------------------------------
    fc = carrier_cycles_per_symbol
    carrier = np.sin(2 * np.pi * fc * t)
    
    # -------------------------------------------------------------
    # 3. M-ASK (Amplitude Shift Keying)
    # -------------------------------------------------------------
    # Amplitudes distribuídas entre 0.3 e 1.0 para manter legibilidade visual
    ask_signal = np.zeros(total_samples)
    for i, s in enumerate(symbols):
        idx = slice(i * samples_per_symbol, (i + 1) * samples_per_symbol)
        t_local = t[idx]
        amp = 0.3 + 0.7 * (s / (M - 1)) if M > 1 else 1.0
        ask_signal[idx] = amp * np.sin(2 * np.pi * fc * t_local)
        
    # -------------------------------------------------------------
    # 4. M-FSK (Frequency Shift Keying)
    # -------------------------------------------------------------
    # Frequências com ciclos inteiros por símbolo para suavidade nas transições
    fsk_signal = np.zeros(total_samples)
    for i, s in enumerate(symbols):
        idx = slice(i * samples_per_symbol, (i + 1) * samples_per_symbol)
        t_local = t[idx] - i  # tempo relativo ao início do símbolo [0, 1)
        
        # Para M=2: 1 ciclo para bit 0, 3 ciclos para bit 1 (idêntico à imagem)
        # Para M>2: frequências crescentes em degraus
        if M == 2:
            freq = 1.0 if s == 0 else 3.0
        else:
            freq = 1.0 + s
            
        fsk_signal[idx] = np.sin(2 * np.pi * freq * t_local)
        
    # -------------------------------------------------------------
    # 5. M-PSK (Phase Shift Keying)
    # -------------------------------------------------------------
    # Fase mapeada uniformemente em [0, 2π)
    psk_signal = np.zeros(total_samples)
    for i, s in enumerate(symbols):
        idx = slice(i * samples_per_symbol, (i + 1) * samples_per_symbol)
        t_local = t[idx]
        
        # Para M=2: bit 1 = 0 rad, bit 0 = π rad (inversão idêntica à imagem)
        if M == 2:
            phase = 0.0 if s == 1 else np.pi
        else:
            phase = 2 * np.pi * s / M
            
        psk_signal[idx] = np.sin(2 * np.pi * fc * t_local + phase)

    # -------------------------------------------------------------
    # Renderização e Estilo
    # -------------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, dpi=300)
    fig.patch.set_facecolor('white')

    plots_data = [
        (t_digital, y_digital, '#f25454', 'Sinal digital\n\n' + ('binário' if M == 2 else f'{M}-ário') , 'step'),
        (t, carrier, '#52c46a', 'Portadora\n de alta frequência', 'line'),
        (t, ask_signal, '#4a69ff', 'Amplitude\nshift\nkeying (ASK)', 'line'),
        (t, fsk_signal, "#4a69ff", 'Frequency\nshift\nkeying (FSK)', 'line'),
        (t, psk_signal, '#4a69ff', 'Phase\nshift\nkeying (PSK)', 'line')
    ]

    label_x_pos = -1.5  # Posição horizontal dos rótulos à esquerda

    for ax, (x, y, color, label, plot_type) in zip(axes, plots_data):
        ax.set_facecolor('white')
        
        # Linha horizontal tracejada passando pelo centro
        ax.plot([-3.0, n_symbols], [0, 0], color='#a5a5a5', linestyle='--', linewidth=1.0, zorder=1)
        
        # Rótulo textual à esquerda com fundo branco para mascarar a linha tracejada
        ax.text(label_x_pos, 0, label, ha='center', va='center', fontsize=11, 
                color='#1a1a1a', linespacing=1.2,
                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='none'))
        
        # Curvas dos sinais
        if plot_type == 'step':
            ax.plot(x, y, color=color, linewidth=1.0, zorder=2)
            # Rótulos dos símbolos digitais no topo
            for i, s in enumerate(symbols):
                ax.text(i + 0.5, 0.45, str(s), ha='center', va='center', fontsize=11, color='#222222')
        else:
            ax.plot(x, y, color=color, linewidth=1.0, zorder=2)
            
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlim(-3.0, n_symbols)
        
        # Remove molduras e marcações dos eixos
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.04, hspace=0.15)
    plt.show()

