# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import sympy as sp
import numpy as np
from numpy.random import normal
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
from utils import symdisp, symplot
import ipywidgets as widgets
from ipywidgets import interact

from optic.comm.modulation import modulateGray, demodulateGray, grayMapping
from optic.dsp.core import firFilter, pulseShape, lowPassFIR, pnorm, upsample
from optic.comm.metrics import signal_power
from optic.plot import eyediagram, pconst

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
#mpl.rcParams['figure.dpi'] = 300
# -

# %load_ext autoreload
# %autoreload 2

# +
#plt.style.use(["science"])
# -

from IPython.core.pylabtools import figsize
figsize(6, 2)

# # Transmissão Digital da Informação

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Canais-limitados-em-banda" data-toc-modified-id="Canais-limitados-em-banda-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Canais limitados em banda</a></span></li><li><span><a href="#Sinalização-para-canais-limitados-em-banda" data-toc-modified-id="Sinalização-para-canais-limitados-em-banda-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Sinalização para canais limitados em banda</a></span><ul class="toc-item"><li><span><a href="#Interferência-intersimbólica-(ISI)" data-toc-modified-id="Interferência-intersimbólica-(ISI)-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Interferência intersimbólica (ISI)</a></span></li><li><span><a href="#Critério-de-Nyquist-para-ausência-de-interferência-intersimbólica" data-toc-modified-id="Critério-de-Nyquist-para-ausência-de-interferência-intersimbólica-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Critério de Nyquist para ausência de interferência intersimbólica</a></span></li><li><span><a href="#Família-de-pulsos-cosseno-levantado-(raised-cosine)" data-toc-modified-id="Família-de-pulsos-cosseno-levantado-(raised-cosine)-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Família de pulsos cosseno levantado (<em>raised cosine</em>)</a></span></li><li><span><a href="#Transmissão-de-sinais-com-resposta-parcial" data-toc-modified-id="Transmissão-de-sinais-com-resposta-parcial-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Transmissão de sinais com <em>resposta parcial</em></a></span></li><li><span><a href="#Detecção-símbolo-a-símbolo-de-sinais-com-ISI-controlada" data-toc-modified-id="Detecção-símbolo-a-símbolo-de-sinais-com-ISI-controlada-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Detecção símbolo-a-símbolo de sinais com ISI controlada</a></span></li></ul></li><li><span><a href="#Detecção-de-sequências-por-máxima-verossimilhança" data-toc-modified-id="Detecção-de-sequências-por-máxima-verossimilhança-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Detecção de sequências por máxima verossimilhança</a></span></li><li><span><a href="#Referências" data-toc-modified-id="Referências-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Referências</a></span></li></ul></div>
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

# <img src="./figuras/Fig14.png" width="800">
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
# y(t)=\sum_{k=-\infty}^{\infty} s_k x(t-kT_s) + v(t)
# \end{equation}
# $$
#
# temos
#
# $$
# \begin{equation}
# y\left(q T_s+\tau_0\right) \equiv y_q=\sum_{k=-\infty}^{\infty} s_k x\left(qT_s- kT_s+\tau_0\right) + v\left(qT_s+\tau_0\right)
# \end{equation}
# $$
#
# em que $\tau_{0}$ é o atraso de propagação do sinal pelo canal. Considerando apenas a representação discreta do sinal, temos 
#
# $$
# \begin{align}
# y_q&=\sum_{k=-\infty}^{\infty} s_k x_{q-k}+v_q, \quad q=0,1, \ldots.\nonumber\\
# &= x_0\left(s_q+\frac{1}{x_0} \sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k}\right)+v_q, \quad q=0,1, \ldots
# \end{align}
# $$
#
# em que $x_0$ é um fator de escala arbitrário que pode ser considerado igual à unidade por conveniência, de modo que
#
# $$
# \begin{equation}\label{ISI_eq1}
# y_q=s_q + \sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k} + v_q, \quad q=0,1, \ldots
# \end{equation}
# $$
#
# Em ($\ref{ISI_eq1}$), temos que $s_q$ é o símbolo transmitido no intervalo de sinalização $q$. Já o termo $\sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k}$ representa a interferência causada em $s_q$ pelos demais símbolos transmitidos, denominada **interferência intersimbólica** (*intersymbol interference* - ISI). Por fim, $v_q$ é uma variável aleatória representado o ruído AWGN no instante de sinalização $q$.
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
h_ch = np.array([0, 0.2, 1, 0.2])
h_ch = h_ch/np.sum(h_ch)

# upsampling
symbolsUp = upsample(symbTx, SpS)
h_ch_Up = upsample(h_ch, SpS)
h_ch_Up = h_ch_Up.real

# pulso NRZ típico
pulse = pulseShape('nrz', SpS)
pulse = pulse/np.sum(pulse)

# formatação de pulso
sigTx = firFilter(pulse, symbolsUp)
sigTx = sigTx/np.max(np.abs(sigTx))

t = np.arange(sigTx.size)*Ta/1e-9

# instantes centrais dos intervalos de sinalização
symbolsUp = upsample(symbTx, SpS)
symbolsUp[symbolsUp==0+1j*0] = np.nan + 1j*np.nan

if constType == 'pam':
    plt.figure(2)
    plt.plot(t, sigTx,'-',linewidth=2)
    plt.plot(t, symbolsUp.real,'ko')
    plt.xlabel('Tempo [ns]')
    plt.ylabel('Amplitude')
    plt.title('Sinal '+str(M)+'-PAM')
    plt.grid()
else:
    plt.figure(2)
    plt.plot(t, sigTx.real,'-',linewidth=3, label = '$Re\{s_n\}$')
    plt.plot(t, symbolsUp.real,'o')
    plt.xlabel('Tempo [ns]')
    plt.ylabel('Amplitude')
    plt.title('Sinal '+str(M)+'-QAM')
    plt.grid()
    
    plt.figure(3)
    plt.plot(t, sigTx.imag,'-',linewidth=3, label = '$Im\{s_n\}$')
    plt.plot(t, symbolsUp.imag,'o')
    plt.xlabel('Tempo [ns]')
    plt.ylabel('Amplitude')
    plt.title('Sinal '+str(M)+'-QAM')
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
h_ch = np.array([0, 0.2, 1, 0.2])
h_ch = np.array([0, 0, 1, 0])

# upsampling
symbolsUp = upsample(symbTx, SpS)
h_ch_Up = upsample(h_ch, SpS)
h_ch_Up = h_ch_Up/np.sum(h_ch_Up)

# pulso NRZ típico
pulse = pulseShape('rrc', SpS, N=4096, alpha=0.005)
pulse = pulse/np.sum(pulse)

# formatação de pulso
sigTx = firFilter(pulse, symbolsUp)
sigTx = pnorm(sigTx)

# canal linear
sigRx = firFilter(h_ch_Up, sigTx)

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
    
# -

pconst(pnorm((sigRx + ruidoC)[2+10*SpS:-10*SpS:SpS]), pType='fast', R=1.5);

# + hide_input=false
# plot PSD
plt.figure(figsize=(6,3));
plt.xlim(-2*Rs/1e6,2*Rs/1e6);
plt.ylim(-90,-10);
plt.psd(sigTx,Fs=Fa/1e6, NFFT = 16*1024, sides='twosided', label = 'Espectro do sinal Tx (entrada do canal)')
plt.psd(firFilter(h_ch_Up, sigTx),Fs=Fa/1e6, NFFT = 16*1024, sides='twosided', label = 'Espectro do sinal Rx (saída do canal)')
plt.xlabel('Frequency [MHz]')
plt.legend(loc='lower left');
# -

# ### Critério de Nyquist para ausência de interferência intersimbólica
#
# Considere que a resposta em frequência $H(f)$ é ideal e limitada em banda, de maneira que
#
# $$
# \begin{equation}
# H(f) = \begin{cases}1, & |f|<B \\ 0, & \text { caso contrário.}\end{cases}.
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
# y_q=s_q + \sum_{\substack{k=-\infty \\ k \neq q}}^{\infty} s_k x_{q-k} + v_q, \quad q=0,1, \ldots,
# \end{equation}
# $$
#
# o que implica que a condição para ausência de ISI é dada por
#
# $$
# \begin{equation}\label{isi_x}
# x(t=n T_s) \equiv x_n= \begin{cases}1, & n=0 \\ 0, & n \neq 0\end{cases}
# \end{equation}
# $$
#
# Esta condição é conhecida como critério de Nyquist para ausência de interferência intersimbólica.
#
# **Teorema (Critério de Nyquist)**: uma condição necessária e suficiente para a ausência de interferência intersimbólica, ou seja, para $x(t)$ satisfazer ($\ref{isi_x}$) é que sua transformada de Fourier $X(f)$ satisfaça
#
# $$
# \sum_{m=-\infty}^{\infty} X(f+m R_s)=T_s.
# $$
#
#
# Prova:
#
# $$
# \begin{equation}
# x(t)=\int_{-\infty}^{\infty} X(f) e^{j 2 \pi f t} d f
# \end{equation}
# $$
#
# Considerando que as amostras são tomadas ao final de cada intervalo de sinalização $t=kT_s$, temos que
# $$
# \begin{equation}
# x(kT_s)=\int_{-\infty}^{\infty} X(f) e^{j 2 \pi f k T_s} d f.
# \end{equation}
# $$
#
# A taxa de sinalização, ou taxa de símbolos, $R_s$ é dada por $R_s = 1/T_s$, de modo que

# $$
# \begin{align}
# x(kT_s) & =\sum_{m=-\infty}^{\infty} \int_{(2 m-1) R_s/2}^{(2 m+1) R_s/2} X(f) e^{j 2 \pi f kT_s} d f\nonumber \\
# & =\sum_{m=-\infty}^{\infty} \int_{-R_s/2}^{R_s/2} X(f+ m R_s) e^{j 2 \pi f kT_s} d f \nonumber\\
# & =\int_{-R_s/2}^{R_s/2}\left[\sum_{m=-\infty}^{\infty} X(f+m R_s)\right] e^{j 2 \pi f kT_s} d f \nonumber\\
# & =\int_{-R_s/2}^{R_s/2} B(f) e^{j 2 \pi f kT_s} d f\label{nyq_1}
# \end{align}
# $$

# Em que a função $B(f)$ é dada por
# $$
# \begin{equation}
# B(f)=\sum_{m=-\infty}^{\infty} X(f+mR_s).
# \end{equation}
# $$
#
# Como $B(f)$ é periódica em $f$ com período $R_s$, esta admite representação em termos de uma série de Fourier
#
# $$
# \begin{align}
# B(f)&=\sum_{k=-\infty}^{\infty} b_k e^{j 2 \pi k f T_s} \nonumber \\
# b_k&=\frac{1}{R_s} \int_{-R_s / 2}^{R_s/ 2} B(f) e^{-\frac{j2 \pi k f}{R_s}} d f = T_s \int_{-R_s / 2}^{R_s/ 2} B(f) e^{-j2 \pi k f T_s} d f \label{nyq_2}
# \end{align}
# $$
#
# Logo, comparando ($\ref{nyq_1}$) e ($\ref{nyq_2}$), temos que
#
# $$
# \begin{equation}
# b_k=T_s x(-k T_s).
# \end{equation}
# $$
#
# Desse modo, na ausência de ISI, os coeficientes $b_k$ devem ser dados por
# $$
# \begin{equation}\label{fourier_coeffs}
# b_k=\begin{cases}T_s, & k=0 \\ 0, & k \neq 0\end{cases}.
# \end{equation}
# $$
#
# o que implica em
#
# $$
# \begin{align}
# B(f)&=T_s\\
# \sum_{m=-\infty}^{\infty} X(f+m R_s)&=T_s.
# \end{align}
# $$

# Supondo que o canal tem banda $B$, teremos então três casos 
#
# 1. Se $R_s > 2B$, as cópias de $X(f)$ espaçadas de $R_s$ no domínio da frequência não se sobrepõem, de modo que não há nenhuma escolha possível de $x(t)$ que permita que $\sum_{m=-\infty}^{\infty} X(f+m R_s)$ seja uma função constante. Como consequência, não existe nenhuma função $x(t)$ que permita a transmissão livre de ISI.
#
# 2. Se $R_s = 2B$, o somatório $\sum_{m=-\infty}^{\infty} X(f+m R_s)$ será uma constante apenas se $X(f)$ for uma função porta
# $$
# \begin{equation}
# X(f)= \begin{cases}T_s & |f|<B \\ 0, & \text { c.c. }\end{cases}
# \end{equation}
# $$
#
# ou seja, se $x(t)$ for uma função sinc
#
# $$
# \begin{equation}
# x(t)=\operatorname{sinc}\left(\frac{t}{T_s}\right).
# \end{equation}
# $$
#
# 3. Se $R_s < 2B$, os temos do somatório $\sum_{m=-\infty}^{\infty} X(f+m R_s)$ se sobrepõem, de modo que existirão diversas escolhas de $x(t)$ que o mesmo seja constante.
#
#
# ### Família de pulsos cosseno levantado (*raised cosine*)
#
# Um conjunto de pulsos que atendem ao critério de Nyquist de ISI nula correponde à família de pulsos cosseno levantado, que pode ser definida no domínio da frequência como
# $$
# \begin{equation}
# X_{r c}(f)= \begin{cases}T_s, & 0 \leq|f| \leq(1-\alpha) / 2 T_s \\ \frac{T_s}{2}\left[1+\cos \frac{\pi T_s}{\alpha}\left(|f|-\frac{1-\alpha}{2 T_s}\right)\right], & \frac{1-\alpha}{2 T_s} \leq|f| \leq \frac{1+\alpha}{2 T_s} \\ 0, & |f|>\frac{1+\alpha}{2 T_s}\end{cases}
# \end{equation}
# $$
# e no domínio do tempo como
# $$
# \begin{align}
# x(t) & =\frac{\operatorname{sen} (\pi t / T_s)}{\pi t / T_s} \frac{\cos (\pi \alpha t / T_s)}{1-4 \alpha^2 t^2 / T_s^2}\nonumber \\
# & =\operatorname{sinc}(t / T_s)\frac{\cos (\pi \alpha t / T_s)}{1-4 \alpha^2 t^2 / T_s^2},
# \end{align}
# $$
#
# em que $0<\alpha<1$ é denominado de parâmetro de *rolloff* ou *excesso de banda*.

# + hide_input=false
Rs = 100e6
Ts = 1/Rs
α = 0.0001
π = sp.pi

f, t = sp.symbols('f, t', real=True)

X = sp.Piecewise((0, ( f  <= -(1 + α)/(2*Ts) ) ), 
                  ((Ts/2)*(1 + sp.cos(π*Ts/α*(sp.Abs(f)-(1-α)/(2*Ts)))), ( f  > -(1 + α)/(2*Ts) )&( f <= -(1-α)/(2*Ts)  ) ) , 
                  (Ts, ( f  > -(1-α)/(2*Ts) )&( f <= (1-α)/(2*Ts))  ) ,
                  ((Ts/2)*(1 + sp.cos(π*Ts/α*(sp.Abs(f)-(1-α)/(2*Ts)))), ( f  > (1-α)/(2*Ts) )&( f <= (1+α)/(2*Ts))  ),
                  (0, ( f  > (1+α)/(2*Ts) ) ) ) 

x = sp.sinc(π*t/Ts)*sp.cos(π*α*t/Ts)/(1-4*α**2*t**2/Ts**2)

finterval = np.arange(-2, 2, 0.01)*(1/Ts)
symplot(f, X, finterval, '$X_{rc}$(f)', xlabel = 'f[Hz]');

tinterval = np.arange(-20, 20, 0.01)*Ts
symplot(t, x, tinterval, '$x_{rc}$(t)', xlabel = 't[s]');
# -

# Duas vantagens importantes:
#
# 1. Devido às características suaves de transição de $x_{rc}(t)$ estes pulsos podem ser implementados na prática por filtros lineares formatadores de pulso.
#
# 2. As bordas laterais dos pulsos $x_{rc}(t)$ convergem para 0 a uma taxa proporcional a $1/t^3$, para $\alpha>0$, diferentemente da função $\operatorname{sinc}(t)$, que convege a $1/t$. Essa proprieadade permite que pequenos desvios do instante de amostragem ótimo façam com que a ISI convirja para um valor finito no caso do pulso $x_{rc}(t)$, enquanto a mesma diverge no caso de pulsos $\operatorname{sinc}(t)$. Assim, pulsos $x_{rc}(t)$ são em geral mais robustos à erros de sincronização do que pulsos $\operatorname{sinc}(t)$.
#
# 3. A complexidade do processamento requerido para a implementação de $x_{rc}(t)$ pode ser dividida entre transmissor e receptor. Considerando a situação ideal em que $$
# \begin{equation}\nonumber
# H(f) = \begin{cases}1, & |f|<B \\ 0, & \text { caso contrário.}\end{cases}
# \end{equation}
# $$Sejam $P_T(f)$ e $P_R(f) = P_T^*(f)$ os filtro formatador de pulso do transmissor e o filtro casado correspondente no receptor, respectivamente, temos que$$
# \begin{align}
# P_T(f)H(f)P_R(f) &= X_{rc}(f)\nonumber\\
# P_T(f)P_R(f) &= X_{rc}(f)\nonumber\\
# |P_T(f)|^2 &= X_{rc}(f)\nonumber\\
# \end{align}
# $$
# ou seja, $X_{rc}(f)$ pode ser alcançada fazendo-se $P_T(f) = P_R^*(f) = \sqrt{X_{rc}(f)} = P_{rrc}(f)$. O pulso cujo espectro é dado por $P_{rrc}(f) = \sqrt{X_{rc}(f)}$ é conhecido como raiz do cosseno levantado (*root raised cosine* - RRC) e é definido no domínio do tempo por
#
# $$
# p_{rrc}(t)= \begin{cases}\frac{1}{T_s}\left(1+\alpha\left(\frac{4}{\pi}-1\right)\right), & t=0 \\ \frac{\alpha}{T_s \sqrt{2}}\left[\left(1+\frac{2}{\pi}\right) \sin \left(\frac{\pi}{4 \alpha}\right)+\left(1-\frac{2}{\pi}\right) \cos \left(\frac{\pi}{4 \alpha}\right)\right], & t= \pm \frac{T_s}{4 \alpha} \\ \frac{1}{T_s} \frac{\sin \left[\pi \frac{t}{T_s}(1-\alpha)\right]+4 \alpha \frac{t}{T_s} \cos \left[\pi \frac{t}{T_s}(1+\alpha)\right]}{\pi \frac{t}{T_s}\left[1-\left(4 \alpha \frac{t}{T_s}\right)^2\right]}, & \text { caso contrário.}\end{cases}
# $$

# ### Transmissão de sinais com *resposta parcial*
#
# Para efeitos práticos, o critério de Nyquist nos indica que é possível projetar formas de pulso para eliminar a presença de ISI no receptor, desde que a taxa de sinalização $R_s$ da transmissão em símbolos/segundo seja tal que $R_s < 2B$, em que $B$ é a banda do canal de comunicações em Hz.
#
# Entretanto, transmitir a uma taxa de $R_s = 2B$ sem ter que recorrer à utilização de pulsos $\operatorname{sinc}(t)$ ainda será possível, desde que uma determinada quantidade controlada de ISI possa ser tolerada no receptor. Por exemplo, suponha que
#
# $$
# x(k T_s)= \begin{cases} 1, & k=0,1 \\ 0, & \text { caso contrário. }\end{cases}
# $$
#
# e então, os coeficientes em ($\ref{fourier_coeffs}$) serão dados por 
#
# $$
# b_k= \begin{cases}T_s & k=0,-1 \\ 0 & \text { caso contrário, }\end{cases}
# $$
#
# de maneira que $B(f)=T_s + T_s e^{j 2 \pi f T_s} = T_s\left(1 + e^{j 2 \pi f T_s}\right)$
#
# Se $R_s > 2B$, o problema continua sem solução. Já no caso de $R_s=2B$, $B(f)$ pode ser alcançado se fizermos 
#
# $$
# \begin{align} 
# X(f) & = \begin{cases}\frac{1}{2 B}\left(1+e^{-j \pi f / B}\right) & |f|<B \\ 0 & \text { caso contrário} \end{cases}\nonumber \\ 
# & = \begin{cases}\frac{1}{B} e^{-j \pi f / 2 B} \cos\left( \frac{\pi f}{2 B}\right) & |f|<B \\ 0 & \text { caso contrário}\end{cases}. \nonumber\end{align}
# $$

# +
Rs = 100e6
Ts = 1/Rs
B = Rs/2

π = sp.pi

f, t = sp.symbols('f, t', real=True)

Bf = Ts + Ts*sp.exp(1j*2*π*f*Ts)

X = sp.Piecewise((1/(2*B)*(1 + sp.exp(1j*π*f/B)), (-B<= f)&(f<= B)), (0, True)) 

finterval = np.arange(-2, 2, 0.01)*2*B
symplot(f, [sp.re(X), sp.im(X)], finterval, ['$\Re{X(f)}$', '$\Im{X(f)}$'], xlabel = 'f[Hz]');
symplot(f, [sp.re(Bf), sp.im(Bf)], finterval, ['$\Re{B(f)}$','$\Im{B(f)}$'], xlabel = 'f[Hz]');
# -

# No domínio do tempo, o pulso associado a $X(f)$ é dado por
#
# $$
# x(t)=\operatorname{sinc}(2 \pi B t)+\operatorname{sinc}\left(2\pi B(t-T_s)\right)
# $$
#
# que é conhecido como *pulso duobinário*.
#

# +
x = sp.sinc(2*B*π*t)+sp.sinc(2*B*π*(t-Ts))

Xabs = sp.piecewise_fold(sp.Abs(X)).simplify()

tinterval = np.arange(-5, 5, 0.01)*Ts
symplot(t, x, tinterval, '$x_{duob}$(t)', xlabel = 't[s]');

finterval = np.arange(-2, 2.01, 0.001)*B
symplot(f, Xabs, finterval, '$|X(f)|$', xlabel = 'f[Hz]');
# -

# Outra possibilidade é fazer
#
# $$
# x(kT_s)= \begin{cases} 1, & k=-1 \\ -1, & k =1\\ 0, & \text { caso contrário. }\end{cases}
# $$
#
# que resultará no pulso
#
# $$
# x(t)=\operatorname{sinc}\left[2\pi B(t+T_s)\right]-\operatorname{sinc}\left[2\pi B(t-T_s)\right]
# $$
#
# cujo espectro é dado por
#
# $$
# X(f)= \begin{cases}\frac{1}{2 B}\left(e^{i \pi f / B}-e^{-j \pi f/ B}\right)=\frac{j}{B} \sin \left(\frac{\pi f}{B}\right) & |f|\leqslant B \\ 0 & |f| > B\end{cases}
# $$

# +
x = sp.sinc(2*B*π*(t+Ts))-sp.sinc(2*B*π*(t-Ts))

X = sp.Piecewise((1/(2*B)*(sp.exp(1j*π*f/B) - sp.exp(-1j*π*f/B)), (-B<= f)&(f<= B)), (0, True)) 

Xabs = sp.piecewise_fold(sp.Abs(X)).simplify()

tinterval = np.arange(-5, 5, 0.0001)*Ts
symplot(t, x, tinterval, '$x_{duob}$(t)', xlabel = 't[s]');

finterval = np.arange(-2, 2.01, 0.001)*B
symplot(f, Xabs, finterval, '$|X(f)|$', xlabel = 'f[Hz]');
# -

# De maneira geral, os pulsos pertencentes ao conjunto de sinais limitados em banda que são definidos como
#
# $$
# x(t)=\sum_{n=-\infty}^{\infty} x\left(\frac{n}{2 B}\right) \operatorname{sinc}\left[2 \pi B\left(t-\frac{n}{2 B}\right)\right]
# $$
#
# e com espectro correspondente
#
# $$
# X(f)= \begin{cases}\frac{1}{2 B} \sum_{n=-\infty}^{\infty} x\left(\frac{n}{2 B}\right) e^{-j n \pi f / B} & |f| \leqslant B \\ 0, & \text{caso contrário.}\end{cases}
# $$
#
# são denominados pulsos ou sinais de *resposta parcial*. A utilização de tais pulsos na transmissão significa que há uma adição controlada de ISI por meio da introdução de duas ou mais amostras não-nulas $x\left(\frac{n}{2 B}\right)$. Tais pulsos permitem a taxa de sinalização da transmissão seja igual a taxa de Nyquist, ou seja, $R_s = 2B$.

# ### Detecção símbolo-a-símbolo de sinais com ISI controlada
#
# Considere uma transmissão duobinária tal que a amostra $y_k$ capturada em cada instante de sinalização $t=kT_s$ seja dada por
#
# $$
# y_k = r_k + v_k = a_k + a_{k-1} + v_k
# $$
#
# em que \{$a_k$\} é a sequência de amplitudes transmitidas e \{$v_k$\} a sequência de amostras de ruído aditivo gaussiano. Considere que $a_k=\pm 1$, com a mesma probabilidade, de modo que $r_k=-2,0,2$ com probabilidades $1/4$, $1/2$ e $1/4$.
#
# Note que, se $a_{k-1}$ for conhecido, o receptor pode simplesmente subtrair seu valor de $r_k$, de modo a eliminar o efeito da ISI. Entretanto, se o receptor cometeu um erro na detecção de $a_{k-1}$, a subtração tenderá a não eliminarar a ISI em $r_k$, podendo até mesmo amplificar o efeito da ISI. Tal fenômeno é conhecido como *propagação de erros*.
#
# A propagação de erros pode ser evitada utilizando-se uma pré-codificação na sequência de bits a ser transmitida.
#
# Considere que a partir de uma sequência de bits $b_k$, uma nova sequência $c_k$ seja gerada fazendo
#
# $$\begin{equation}\label{precod}
# c_k = b_k \ominus c_{k-1}, \quad k = 1, 2, \ldots
# \end{equation}
# $$
#
# em que $\ominus$ representa a operação de subtração módulo-2 (que é idêntica à operação $\oplus$ adição módulo-2, ou XOR). 
#
# Considerando que a modulação utilizada é a 2-PAM antipodal, temos que a sequência correspondente de amplitudes da modulação é dada por
#
# $$
# a_k = 2c_k -1.
# $$
#
# Consequentemente, a sequência de amplitudes do sinal *duobinário* produzido a partir de $a_k$ é dada por
#
# $$
# r_k = a_k + a_{k-1}.
# $$
#
# Temos então que
#
# $$\begin{align}
# r_k &= a_k + a_{k-1}\nonumber\\
# &= (2c_k -1) + (2c_{k-1} -1)\nonumber\\
# &= 2(c_k + c_{k-1}-1)\nonumber\\
# \end{align}$$
#
# ou seja,
#
# $$
# c_k + c_{k-1} = \frac{r_k}{2} + 1
# $$
#
# De ($\ref{precod}$) temos que $c_k \oplus c_{k-1} = b_k$, de modo que 
#
# $$
# b_k=\frac{r_k}{2} + 1 (\bmod 2)
# $$

# +
# gera sequência pseudo-aleatória de bits
bk = np.random.randint(2, size = 20)

ck = np.zeros(bk.shape, dtype=int)
rk = np.zeros(bk.shape, dtype=int)
dk = np.zeros(bk.shape, dtype=int)

for k in range(len(bk)):
    if k == 0:
        ck[k] = bk[k]
    else:
        ck[k] = bk[k]^ck[k-1]

ak = 2*ck-1

for k in range(len(bk)):
    if k == 0:
        rk[k] = ak[k]
    else:
        rk[k] = ak[k] + ak[k-1]
        
        
for k in range(len(dk)):
    dk[k] = (rk[k]/2 + 1)%2
    
print(f'Bits (bk):{bk}\n')
print(f'Bits após pré-codificação (ck):{ck}\n')
print(f'Amplitudes sinal 2-PAM (ak):{ak}\n')
print(f'Amplitudes sinal 2-PAM duobinário (rk) :{rk}\n')
print(f'Bits após decodificação (dk) :{dk}\n')


# -

# A regra de decisão no receptor torna-se, então
#
# $$
# b_k = \begin{cases}0, & r_k = +2, -2\\
# 1, & r_k = 0.
# \end{cases}
# $$
#
# Com a presença de ruído, as amostras observadas pelo receptor são dadas por $y_k = r_k + v_k$, de modo que a decisão pode ser tomada baseada na comparação do sinal recebido com dois limiares de decisão posicionados em +1 e -1, ou seja
#
# $$
# b_k= \begin{cases}1, & \text { se }-1<y_k<1 \\ 0, & \text { se }\left|y_k\right| \geq 1\end{cases}
# $$
#

# +
def duob(SpS=2, Nsamples=16, reverse=False, alpha=0.01):    
       
    p = pulseShape('rrc', SpS, N=Nsamples+2*SpS, alpha=alpha)
    
    x = p + np.roll(p, -SpS)
    
    if reverse:
        indrev = np.arange(len(p)-1, -1, -1)
        x = x[indrev]        
        
    return x[SpS:-SpS]

# x = duob(16, 1024, reverse=True)
# plt.plot(x, '-')
# plt.plot(p[400:600], '--')
# +
M = 2
constType = 'pam'

# parâmetros da simulação
SpS = 16            # Amostras por símbolo
Rs  = 100e6         # Taxa de símbolos
Ts  = 1/Rs          # Período de símbolo em segundos
Fa  = 1/(Ts/SpS)    # Frequência de amostragem do sinal (amostras/segundo)
Ta  = 1/Fa          # Período de amostragem
rolloff = 0.005

# gera sequência pseudo-aleatória de bits
bitsTx = np.random.randint(2, size = int(100000*np.log2(M)))
cbitsTx = np.zeros(bitsTx.shape, dtype=int)

# pré-codificação
for k in range(len(bitsTx)):
    if k == 0:
        cbitsTx[k] = bitsTx[k]
    else:
        cbitsTx[k] = bitsTx[k]^cbitsTx[k-1]

# gera sequência de símbolos modulados
symbTx = modulateGray(cbitsTx, M, constType)    
symbTx = pnorm(symbTx) # power normalization

# upsampling
symbolsUp = upsample(symbTx, SpS)

# pulso duobinário
pulseDoub = duob(SpS, Nsamples=2048, reverse=False, alpha=rolloff)
pulse = pulseShape('rrc', SpS, N=2048, alpha=rolloff) 
pulse = pulse/max(abs(pulse))

# formatação de pulso
sigTx = firFilter(pulseDoub, symbolsUp)

t = np.arange(sigTx.size)*Ta/1e-9

# filtro casado
sigRx = firFilter(pulse, sigTx)
sigRx = pnorm(sigRx)

# ruído gaussiano branco
Namostras = sigTx.size
σ2  = 0.00025 # variância
μ   = 0         # média

σ      = np.sqrt(σ2*SpS)
ruido  = normal(μ, σ, Namostras)
ruidoC  = (normal(μ, σ, Namostras) + 1j*normal(μ, σ, Namostras))/np.sqrt(2)

pconst(pnorm((sigRx + ruidoC)[2:-10*SpS:SpS]), pType='fast', R=2.5);

lmax = np.max(sigRx.real)
lmin = np.min(sigRx.real)

# diagrama de olho
Nsamples = sigTx.size
if constType == 'pam':
    eyediagram(sigTx.real, Nsamples, SpS, plotlabel= str(M)+'-PAM', ptype='fancy')
    eyediagram((sigRx + ruido).real, Nsamples, SpS, plotlabel= str(M)+'-PAM', ptype='fancy')
    sigRx = sigRx + ruido
else:
    eyediagram(sigTx, Nsamples, SpS, plotlabel= str(M)+'-QAM', ptype='fancy')
    eyediagram(sigRx + ruidoC, Nsamples, SpS, plotlabel= str(M)+'-QAM', ptype='fancy')
    sigRx = sigRx + ruidoC   


# detecção
sigRx = sigRx[2::SpS].real

bitsRx = np.ones(bitsTx.shape, dtype=int)
bitsRx[sigRx>=lmax/2] = 0
bitsRx[sigRx<=lmin/2] = 0

discard = 100

BER = np.mean(np.roll(bitsRx,1)[discard:-discard]^bitsTx[discard:-discard])
print(f'BER = {BER:.2e}')
# -

# ## Detecção de sequências por máxima verossimilhança
#
# Considere o caso em que sequência de sinais recebidos não está sujeita a nenhum efeito de memória, ou seja, o vetor $\mathbf{r}_k$ observado no intervalo de sinalização $k$ é *estatisticamente independente* dos vetores $\left\lbrace \mathbf{r}_{k'} \right\rbrace_{k'=-\infty}^{\infty}$ e $k' \neq k$, observados nos demais intervalos de sinalização. Neste caso, o detector que opera *símbolo-a-símbolo*, ou seja, que decide utilizando apenas a informação $P(\mathbf{s}_m|\mathbf{r}_k)$ presente no intervalo de sinalização $k$ é o detector ótimo no sentido de minimização da probabilidade de erro.
#
# Entretanto, na presença de memórioa, ou seja, caso exista *dependência estatística* entre a sequência de sinais $\left\lbrace \mathbf{r}_{k} \right\rbrace_{k=-\infty}^{\infty}$, o detector ótimo para o sinal recebido no intervalo de sinalização $k$ deverá levar em conta não apenas este intervalo de sinalização, mas uma sequência de intervalos de sinalização consecutivos.
#
# Como exemplo, considere o caso da transmissão duobinária 2-PAM. Seja $L$ o intervalo de mémoria, em intervalos de sinalização, presente na transmissão, para um sinal duobinário temos que $L=1$.

# $$
# \begin{aligned}
# \begin{array}{|c|c|c|c|}
#   \hline
#   a_{k-1} & a_{k} & r_{k} = a_k + a_{k-1}  \\
#   \hline
#   -1 & -1 & -2 \\
#   \hline
#   1 & -1 & 0  \\
#   \hline
#   -1 & 1 & 0 \\
#   \hline
#   1 & 1 & 2 \\
#   \hline
# \end{array}
# \end{aligned}
# $$

# <img src="./figuras/Fig13.png" width="600">
# <center>Fig.10: Treliça representando a evolução de estados da modulação 2-PAM duobinário.</center>

# Considere um vetor $\boldsymbol{y}$ composto pela sequência de amostras da saída do filtro do receptor sobre $N$ intervalos de sinalização $\boldsymbol{y} = [y_1, y_2, \dots, y_N]^T$. Temos que
#
#
# $$\boldsymbol{y} = \boldsymbol{r} + \boldsymbol{v}$$
#
#
# em que $\boldsymbol{r} = [r_1, r_2, \dots, r_N]^T$ é a sequência de amplitudes do sinal duobinário livres de ruído, $\boldsymbol{v} = [v_1, v_2, \dots, v_N]^T$ é a sequência de amostras de ruído gaussiano, i.e. $\boldsymbol{v}\sim \mathcal{N}(\mathbf{0}, \Sigma_v)$.
#
# Seja  $\boldsymbol{a} = [a_1, a_2, \dots, a_N]^T$ a sequência de amplitudes binárias transmitidas, a fdp condicional $p(\boldsymbol{y}|\boldsymbol{a})$ pode ser escrita como
#
# $$
# p(\boldsymbol{y}|\boldsymbol{a}) = \frac{1}{(2\pi |\Sigma_v|)^{N/2}} \exp \left[-\frac{1}{2} (\boldsymbol{y}-\boldsymbol{r})^T \Sigma_v^{-1} (\boldsymbol{y}-\boldsymbol{r}) \right].
# $$
#
# Desse modo, temos que a probabilidade da sequência $\boldsymbol{a}$ ter sido transmitida condicionada à sequência de observações $\boldsymbol{y}$ pode ser escrita como
# $$
# P(\boldsymbol{a}|\boldsymbol{y}) = \frac{p(\boldsymbol{y}|\boldsymbol{a})P(\boldsymbol{a})}{p(\boldsymbol{y})} \propto p(\boldsymbol{y}|\boldsymbol{a})P(\boldsymbol{a})
# $$
#
# Assumindo que o detector decidirá por uma sequência $\boldsymbol{\hat{a}}$, a decisão ótima será tal que
#
# $$\boldsymbol{\hat{a}} = \underset{\boldsymbol{a}}{\operatorname{argmax}} p(\boldsymbol{y}|\boldsymbol{a})P(\boldsymbol{a})$$.
#
# correspondendo ao critério MAP.
#
# Considerando que todas as sequências são equiprováveis
#
# $$
# P(\boldsymbol{a}|\boldsymbol{y}) \propto p(\boldsymbol{y}|\boldsymbol{a})
# $$
#
# de modo que
#
# $$\boldsymbol{\hat{a}} = \underset{\boldsymbol{a}}{\operatorname{argmax}} p(\boldsymbol{y}|\boldsymbol{a})$$,
#
# que corresponde ao critério de decisão ML.
#
# $$
# \ln p(\boldsymbol{y}|\boldsymbol{a}) = -\frac{N}{2}\ln(2\pi |\Sigma_v|) -\frac{1}{2} (\boldsymbol{y}-\boldsymbol{r})^T \Sigma_v^{-1} (\boldsymbol{y}-\boldsymbol{r})
# $$
#
# Maximizar $\ln p(\boldsymbol{y}|\boldsymbol{a})$ equivale a encontrar a sequência $\boldsymbol{a}$ que minimiza $(\boldsymbol{y}-\boldsymbol{r})^T \Sigma_v^{-1} (\boldsymbol{y}-\boldsymbol{r})$
#
# $$\boldsymbol{\hat{a}} = \underset{\boldsymbol{a}}{\operatorname{argmin}} (\boldsymbol{y}-\boldsymbol{r})^T \Sigma_v^{-1} (\boldsymbol{y}-\boldsymbol{r})$$,
#
# como $r_k =x_0a_k + x_1a_{k-1} + \dots =  \sum_{q=0}^{L}x_qa_{k-q}$
#
# $$\boldsymbol{\hat{a}} = \underset{\boldsymbol{a}}{\operatorname{argmin}} \left[ \sum_{m=1}^{N}\left(y_m - \sum_{q=0}^{L}x_qa_{m-q}\right)^2\right]$$,

# ## Referências
#
# [1] J. G. Proakis, M. Salehi, Communication Systems Engineering, 2nd Edition, Pearson, 2002.
