# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import sympy as sp
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display, Math
from IPython.display import display as disp
from utils import symdisp, symplot
import ipywidgets as widgets
from ipywidgets import interact

from commpy.utilities import upsample

from optic.modulation import modulateGray, demodulateGray, GrayMapping
from optic.dsp import firFilter, pulseShape, lowPassFIR, pnorm, sincInterp
from optic.metrics import signal_power
from optic.plot import eyediagram, pconst
# -

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.options.display.float_format = '{:,d}'.format

# +
from IPython.core.display import HTML
from IPython.core.pylabtools import figsize
from IPython.display import display

HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
# -

# %load_ext autoreload
# %autoreload 2

figsize(8, 3)

# # Transmissão Digital da Informação

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Canais-limitados-em-banda" data-toc-modified-id="Canais-limitados-em-banda-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Canais limitados em banda</a></span></li><li><span><a href="#Sinalização-para-canais-limitados-em-banda" data-toc-modified-id="Sinalização-para-canais-limitados-em-banda-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Sinalização para canais limitados em banda</a></span><ul class="toc-item"><li><span><a href="#Interferência-intersimbólica-(ISI)" data-toc-modified-id="Interferência-intersimbólica-(ISI)-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Interferência intersimbólica (ISI)</a></span></li><li><span><a href="#Critério-de-Nyquist-para-ausência-de-interferência-intersimbólica" data-toc-modified-id="Critério-de-Nyquist-para-ausência-de-interferência-intersimbólica-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Critério de Nyquist para ausência de interferência intersimbólica</a></span></li></ul></li><li><span><a href="#Referências" data-toc-modified-id="Referências-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Referências</a></span></li></ul></div>
# -

# # Receptores ótimos para canais AWGN

# <img src="./figuras/Fig1.png" width="900">
#  <center>Fig.1: Diagrama de blocos de um sistema de transmissão digital genérico.</center>

# ##  Canais limitados em banda

# Na maioria das situações práticas, a transmissão digital deve ser estabelecida por meio de um canal cuja banda de frequências disponível para comunicação é limitada. Desse modo, existirão restrições sobre o conteúdo de frequências que os sinais transmitidos e recebidos poderão conter. 
#
# Na análise que segue, considere que o canal de comunicações pode ser modelado como um sistema linear com uma resposta ao impulso $h(t)$ e resposta em frequência $H(f)$ limitada a uma banda de $B$ Hz, de modo que 
#
# $$
# \begin{equation}
# H(f) = \begin{cases}|H(f)|e^{\theta(f)}, & |f|<B \\ 0, & \text { caso contrário.}\end{cases}
# \end{equation}
# $$
#
# e $H(f) = \int_{-\infty}^{\infty}h(t)e^{-2\pi f t} dt$, $|H(f)|$ é a resposta de amplitude e $\theta(f)$ a resposta de fase do canal. A partir da resposta de fase podemos definir o *atraso de grupo* como 
#
# $$
# \begin{equation}
# \tau(f)=-\frac{1}{2 \pi} \frac{d \theta(f)}{d f}.
# \end{equation}
# $$
#
# O atraso de grupo corresponde ao intervalo de tempo com que cada componente de frequência do sinal transmitido atravessa o canal linear. Um canal linear não causará distorção nos sinais por ele transmitidos se, por todo o espectro do sinal transmitido, $|H(f)|$ for constante e $\theta(f)$ for uma função linear de $f$, ou seja, se o atraso de grupo for constante.

# ## Sinalização para canais limitados em banda
#
# Assuma que em cada intervalo de sinalização $T_s$, o transmissor envia um sinal $s_m(t)$ dentre os $M$ possíveis do esquema de modulação utilizado, i.e. $\left\lbrace s_m(t), m = 1,2,\dots, M\right\rbrace$. Considere que no intervalo de $0\leq t \leq T_s$ o transmissor enviou a sinal $s_m(t)$. Após a filtragem do canal os sinais transmitidos são afetados apenas por AWGN, como ilustrado na Fig. 2. 

# <img src="./figuras/Fig12.png" width="600">
#
# <center>Fig.2: Esquemático de um sistema de transmissão digital via canal linear e AWGN.</center>

# Desse modo, o sinal $r(t)$ que chega a entrada do receptor pode ser representado por
#
#
# $$
# \begin{align}
# r(t) & =\left[\sum_{k=-\infty}^{\infty} s_k p\left(t-k T_s\right)\right] \ast h(t) + n(t)\nonumber \\
# & =\sum_{k=-\infty}^{\infty} s_k p(t)\ast \delta\left(t-k T_s\right) \ast h(t) + n(t)\nonumber \\
# & =\sum_{k=-\infty}^{\infty} s_k p(t)\ast h(t)\ast\delta\left(t-k T_s\right)  + n(t)\nonumber \\
# & =\sum_{k=-\infty}^{\infty} s_k g(t)\ast\delta\left(t-k T_s\right)  + n(t)\nonumber \\
# & =\sum_{k=-\infty}^{\infty} s_k g\left(t-k T_s\right)  + n(t) \label{ch_model_1}\\
# \end{align}
# $$
#
# em que $g(t)$ é o pulso resultante da convolução de $p(t)$ com a resposta ao impulso do canal $h(t)$.
#
# ### Interferência intersimbólica (ISI)
#
# Assuma que no receptor o sinal $r(t)$ é filtrado e amostrado nos instantes $t=qT_s + \tau_{0}, q=0, 1, \ldots$. Seja a saída do filtro do receptor dada por
#
# $$
# \begin{equation}
# y(t)=\sum_{k=-\infty}^{\infty} s_k x(t-kT_s) + u(t)
# \end{equation}
# $$
#
# temos
#
# $$
# \begin{equation}
# y\left(q T_s+\tau_0\right) \equiv y_q=\sum_{k=-\infty}^{\infty} s_k x\left(qT_s- kT_s+\tau_0\right) + u\left(qT_s+\tau_0\right)
# \end{equation}
# $$
#
# em que $\tau_{0}$ é o atraso de propagação do sinal pelo canal. Considerando apenas a representação discreta do sinal, temos 
#
# $$
# \begin{align}
# y_q&=\sum_{k=-\infty}^{\infty} s_k x_{q-k}+u_q, \quad q=0,1, \ldots.\nonumber\\
# &= x_0\left(s_q+\frac{1}{x_0} \sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k}\right)+u_q, \quad q=0,1, \ldots
# \end{align}
# $$
#
# em que $x_0$ é um fator de escala arbitrário que pode ser considerado igual à unidade por conveniência, de modo que
#
# $$
# \begin{equation}\label{ISI_eq1}
# y_q=s_q + \sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k} + u_q, \quad q=0,1, \ldots
# \end{equation}
# $$
#
# Em ($\ref{ISI_eq1}$), temos que $s_q$ é o símbolo transmitido no intervalo de sinalização $q$. Já o termo $\sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k}$ representa a interferência causada em $s_q$ pelos demais símbolos transmitidos, denominada **interferência intersimbólica** (*intersymbol interference* - ISI). Por fim, $u_q$ é uma variável aleatória representado o ruído AWGN no instante de sinalização $q$.
#

# +
M = 4
constType = 'qam'

# parâmetros da simulação
SpS = 16            # Amostras por símbolo
Rs  = 100e6         # Taxa de símbolos
Ts  = 1/Rs          # Período de símbolo em segundos
Fa  = 1/(Ts/SpS)    # Frequência de amostragem do sinal (amostras/segundo)
Ta  = 1/Fa          # Período de amostragem

# gera sequência pseudo-aleatória de bits
bitsTx = np.random.randint(2, size = int(25*np.log2(M)))

# gera sequência de símbolos modulados
symbTx = modulateGray(bitsTx, M, constType)    
symbTx = pnorm(symbTx) # power normalization
symbTx = np.insert(symbTx,0, 0)

# resposta do canal linear
h_ch = np.array([0, 0.1, 1, 0.1])

# upsampling
symbolsUp = upsample(symbTx, SpS)
h_ch_Up = upsample(h_ch, SpS)
h_ch_Up = h_ch_Up.real

# pulso NRZ típico
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# formatação de pulso
sigTx = firFilter(pulse, symbolsUp)

t = np.arange(sigTx.size)*Ta/1e-9

# instantes centrais dos intervalos de sinalização
symbolsUp = upsample(symbTx, SpS)
symbolsUp[symbolsUp==0+1j*0] = np.nan + 1j*np.nan

if constType == 'pam':
    plt.figure(2)
    plt.plot(t, sigTx,'-',linewidth=2)
    plt.plot(t, symbolsUp.real,'ko')
    plt.xlabel('tempo [ns]')
    plt.ylabel('amplitude')
    plt.title('sinal '+str(M)+'-PAM')
    plt.grid()
else:
    plt.figure(2)
    plt.plot(t, sigTx.real,'-',linewidth=3, label = '$Re\{s_n\}$')
    plt.plot(t, symbolsUp.real,'o')
    plt.xlabel('tempo [ns]')
    plt.ylabel('amplitude')
    plt.title('sinal '+str(M)+'-QAM')
    plt.grid()
    
    plt.figure(3)
    plt.plot(t, sigTx.imag,'-',linewidth=3, label = '$Im\{s_n\}$')
    plt.plot(t, symbolsUp.imag,'o')
    plt.xlabel('tempo [ns]')
    plt.ylabel('amplitude')
    plt.title('sinal '+str(M)+'-QAM')
    plt.grid()

# canal linear
sigRx = firFilter(h_ch_Up, sigTx)

# ruído gaussiano branco
Namostras = sigTx.size
σ2  = 1e-6  # variância
μ   = 0      # média

σ      = np.sqrt(σ2) 
ruido  = normal(μ, σ, Namostras)
ruidoC  = (normal(μ, σ, Namostras) + 1j*normal(μ, σ, Namostras))/np.sqrt(2)

t = np.arange(sigTx.size)*Ta/1e-9
if constType == 'pam':
    plt.figure(2)
    plt.plot(t, (sigRx + ruido).real,'r--',alpha=1, linewidth=2)
    t = (0.5*Ts + np.arange(0, symbTx.size+1)*Ts)/1e-9
    plt.vlines(t, min(symbTx), max(symbTx), linestyles='dashed', color = 'k');
    plt.xlim(min(t), max(t));
else:
    plt.figure(2)
    plt.plot(t, (sigRx + ruidoC).real,'r--',alpha=1, linewidth=2)
    plt.plot(t, symbolsUp.real,'o')
    plt.figure(3)
    plt.plot(t, (sigRx + ruidoC).imag,'r--',alpha=1, linewidth=2)
    plt.plot(t, symbolsUp.imag,'o')
    t = (0.5*Ts + np.arange(0, symbTx.size+1)*Ts)/1e-9
    plt.figure(2)
    plt.vlines(t, min(symbTx), max(symbTx), linestyles='dashed', color = 'k');
    plt.xlim(min(t), max(t));
    plt.figure(3)
    plt.vlines(t, min(symbTx), max(symbTx), linestyles='dashed', color = 'k');
    plt.xlim(min(t), max(t));

# +
# generate pseudo-random bit sequence
bitsTx = np.random.randint(2, size = int(250000*np.log2(M)))

# generate ook modulated symbol sequence
symbTx = modulateGray(bitsTx, M, constType)    
symbTx = pnorm(symbTx) # power normalization

# resposta do canal linear
h_ch = np.array([0, 0.1, 1, 0.1])

# upsampling
symbolsUp = upsample(symbTx, SpS)
h_ch_Up = upsample(h_ch, SpS)
h_ch_Up = h_ch_Up

# pulso NRZ típico
pulse = pulseShape('nrz', SpS)
pulse = pulse/max(abs(pulse))

# formatação de pulso
sigTx = firFilter(pulse, symbolsUp)
sigTx = pnorm(sigTx)

# canal linear
sigRx = firFilter(h_ch_Up, sigTx)
sigRx = pnorm(sigRx)

# ruído gaussiano branco
Namostras = sigTx.size
σ2  = 1e-7 # variância
μ   = 0         # média

σ      = np.sqrt(σ2*SpS)
ruido  = normal(μ, σ, Namostras)
ruidoC  = (normal(μ, σ, Namostras) + 1j*normal(μ, σ, Namostras))/np.sqrt(2)

# diagrama de olho
Nsamples = sigTx.size
if constType == 'pam':
    eyediagram(sigTx.real, Nsamples, SpS, plotlabel= str(M)+'-PAM', ptype='fancy')
    eyediagram((sigRx + ruido).real, Nsamples, SpS, plotlabel= str(M)+'-PAM', ptype='fancy')
else:
    eyediagram(sigTx, Nsamples, SpS, plotlabel= str(M)+'-QAM', ptype='fancy')
    eyediagram(sigRx + ruidoC, Nsamples, SpS, plotlabel= str(M)+'-QAM', ptype='fancy')
    
pconst(pnorm((sigRx + ruidoC)[10*SpS:-10*SpS:SpS]), pType='fast', R=1.5);

# + hide_input=false
# plot PSD
plt.figure();
plt.xlim(-2*Rs,2*Rs);
plt.ylim(-250,-20);
plt.psd(sigTx,Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Espectro do sinal Tx (entrada do canal)')
plt.psd(pnorm(firFilter(pulse, sigRx)),Fs=Fa, NFFT = 16*1024, sides='twosided', label = 'Espectro do sinal Rx (saída do canal)')
plt.legend(loc='upper left');
# -

# ### Critério de Nyquist para ausência de interferência intersimbólica
#
# Considere que a resposta em frequência $H(f)$ é ideal e limitada em banda, de maneira que
#
# $$
# \begin{equation}
# |H(f)| = \begin{cases}1, & |f|<B \\ 0, & \text { caso contrário.}\end{cases}.
# \end{equation}
# $$
#
# Logo, assumindo que o receptor aplica um filtro casado ao pulso $p(t)$ transmitido, temos que $X(f) = |P(f)|^2$, em que
#
# $$
# \begin{equation}
# x(t) = \int_{-B}^{B}X(f)e^{2\pi f t} df.
# \end{equation}
# $$
#
# Desejamos encontrar as condições que $p(t)$ deve atender e, por consequência, $x(t)$, para que a interferência intersimbólica seja nula. Sabemos que
#
# $$
# \begin{equation}\label{ISI_eq2}
# y_q=s_q + \sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k} + u_q, \quad q=0,1, \ldots,
# \end{equation}
# $$
#
# o que implica que a condição para ausência de ISI é dada por
#
# $$
# \begin{equation}
# x(t=q T_s) \equiv x_q= \begin{cases}1, & q=0 \\ 0, & q \neq 0\end{cases}
# \end{equation}
# $$
#
# Esta condição é conhecida como critério de Nyquist para ausência de interferência intersimbólica.
#
# Prova:
#
# $$
# \begin{equation}
# x(t)=\int_{-\infty}^{\infty} X(f) e^{j 2 \pi f t} d f
# \end{equation}
# $$
#
# $$
# \begin{equation}
# x(kT_s)=\int_{-\infty}^{\infty} X(f) e^{j 2 \pi f k T_s} d f
# \end{equation}
# $$
#
# $$
# \begin{align}
# x(kT_s) & =\sum_{m=-\infty}^{\infty} \int_{(2 m-1) / 2 T_s}^{(2 m+1) / 2 T_s} X(f) e^{j 2 \pi f kT_s} d f \\
# & =\sum_{m=-\infty}^{\infty} \int_{-1 / 2 T_s}^{1 / 2 T_s} X(f+ m / T_s) e^{j 2 \pi f kT_s} d t \\
# & =\int_{-1 / 2 T_s}^{1 / 2 T_s}\left[\sum_{m=-\infty}^{\infty} X(f+m / T_s)\right] e^{j 2 \pi f kT_s} d f \\
# & =\int_{-1 / 2 T_s}^{1 / 2 T_s} B(f) e^{j 2 \pi f kT_s} d f
# \end{align}
# $$
#
# $$
# \begin{equation}
# B(f)=\sum_{m=-\infty}^{\infty} X(f+m/T_s)
# \end{equation}
# $$
#
# $$
# \begin{align}
# B(f)&=\sum_{k=-\infty}^{\infty} b_k e^{j 2 \pi k f T_s} \\
# b_k&=T \int_{-1 / 2 T_s}^{1 / 2 T_s} B(f) e^{-j 2 \pi k f T_s} d f
# \end{align}
# $$
#
# $$
# \begin{equation}
# b_k=T_s x(-k T_s)
# \end{equation}
# $$
#
# $$
# \begin{equation}
# b_k=\begin{cases}T_s, & k=0 \\ 0, & k \neq 0\end{cases}.
# \end{equation}
# $$
#
# $$
# \begin{align}
# B(f)&=T_s\\
# \sum_{m=-\infty}^{\infty} X(f+m / T_s)&=T_s
# \end{align}
# $$

# ## Referências
#
# [1] J. G. Proakis, M. Salehi, Communication Systems Engineering, 2nd Edition, Pearson, 2002.
