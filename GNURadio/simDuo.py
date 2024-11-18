#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Transmission with doubinary
# Author: Edson Porto da Silva
# GNU Radio version: 3.9.4.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import channels
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore



from gnuradio import qtgui

class simDuo(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Transmission with doubinary", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Transmission with doubinary")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "simDuo")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000
        self.SamplesPerSymbol = SamplesPerSymbol = 16
        self.Constellation = Constellation = digital.constellation_qpsk().base()
        self.rolloff = rolloff = 0.005
        self.preconv = preconv = digital.adaptive_algorithm_cma( Constellation, 0.005, 2).base()
        self.numTaps = numTaps = 75
        self.noiseStd = noiseStd = 0.01
        self.delay = delay = 2
        self.chBW = chBW = 0.75*samp_rate/SamplesPerSymbol
        self.AdaptiveEqualizer = AdaptiveEqualizer = digital.adaptive_algorithm_lms( Constellation, 0.00005).base()

        ##################################################
        # Blocks
        ##################################################
        self._noiseStd_range = Range(0, 10, 0.01, 0.01, 200)
        self._noiseStd_win = RangeWidget(self._noiseStd_range, self.set_noiseStd, "Noise std", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._noiseStd_win)
        self._delay_range = Range(0, 32, 1, 2, 200)
        self._delay_win = RangeWidget(self._delay_range, self.set_delay, "'delay'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._delay_win)
        self._chBW_range = Range(10, samp_rate/2, 10, 0.75*samp_rate/SamplesPerSymbol, 200)
        self._chBW_win = RangeWidget(self._chBW_range, self.set_chBW, "channel BW", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._chBW_win)
        self.qtgui_time_sink_x_1 = qtgui.time_sink_c(
            numTaps, #size
            samp_rate/SamplesPerSymbol, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(False)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_axis_labels(True)
        self.qtgui_time_sink_x_1.enable_control_panel(False)
        self.qtgui_time_sink_x_1.enable_stem_plot(True)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [0, 1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_1.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_1.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_time_sink_x_1_win, 2, 0, 2, 3)
        for r in range(2, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_1 = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            "", #name
            3,
            None # parent
        )
        self.qtgui_freq_sink_x_1.set_update_time(0.10)
        self.qtgui_freq_sink_x_1.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_1.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_1.enable_autoscale(False)
        self.qtgui_freq_sink_x_1.enable_grid(False)
        self.qtgui_freq_sink_x_1.set_fft_average(0.1)
        self.qtgui_freq_sink_x_1.enable_axis_labels(True)
        self.qtgui_freq_sink_x_1.enable_control_panel(False)
        self.qtgui_freq_sink_x_1.set_fft_window_normalized(False)



        labels = ['Tx signal', 'Tx signal + channel response', 'Tx signal + channel response + noise', '', '',
            '', '', '', '', '']
        widths = [2, 2, 2, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "black", "red", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(3):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_1.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_1.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_1.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_1.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_1.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_1_win = sip.wrapinstance(self.qtgui_freq_sink_x_1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_1_win, 0, 0, 2, 3)
        for r in range(0, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 3):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_eye_sink_x_0 = qtgui.eye_sink_c(
            1024*16, #size
            samp_rate, #samp_rate
            1, #number of inputs
            None
        )
        self.qtgui_eye_sink_x_0.set_update_time(0.10)
        self.qtgui_eye_sink_x_0.set_samp_per_symbol(SamplesPerSymbol)
        self.qtgui_eye_sink_x_0.set_y_axis(-2, 2)

        self.qtgui_eye_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_eye_sink_x_0.enable_tags(False)
        self.qtgui_eye_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_eye_sink_x_0.enable_autoscale(True)
        self.qtgui_eye_sink_x_0.enable_grid(False)
        self.qtgui_eye_sink_x_0.enable_axis_labels(True)
        self.qtgui_eye_sink_x_0.enable_control_panel(False)


        labels = ['I(t)', 'Q(t)', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'blue', 'blue', 'blue', 'blue',
            'blue', 'blue', 'blue', 'blue', 'blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [3, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_eye_sink_x_0.set_line_label(i, "Eye [Re{{Data {0}}}]".format(round(i/2)))
                else:
                    self.qtgui_eye_sink_x_0.set_line_label(i, "Eye [Im{{Data {0}}}]".format(round((i-1)/2)))
            else:
                self.qtgui_eye_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_eye_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_eye_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_eye_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_eye_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_eye_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_eye_sink_x_0_win = sip.wrapinstance(self.qtgui_eye_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_eye_sink_x_0_win, 0, 3, 2, 3)
        for r in range(0, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 6):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            1024, #size
            "", #name
            2, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self.qtgui_const_sink_x_0.set_y_axis(-2.5, 2.5)
        self.qtgui_const_sink_x_0.set_x_axis(-2.5, 2.5)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(True)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['Received constellation', 'After linear equalization', 'After CPR', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "black", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_const_sink_x_0_win, 2, 3, 2, 3)
        for r in range(2, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(3, 6):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                chBW,
                10,
                window.WIN_HAMMING,
                6.76))
        self.fir_filter_xxx_0 = filter.fir_filter_ccc(2, [1, 0])
        self.fir_filter_xxx_0.declare_sample_delay(0)
        self.filter_fft_rrc_filter_0_0 = filter.fft_filter_ccc(1, firdes.root_raised_cosine(1, samp_rate, samp_rate//SamplesPerSymbol, rolloff, 2048), 1)
        self.filter_fft_rrc_filter_0 = filter.fft_filter_ccc(SamplesPerSymbol//2, firdes.root_raised_cosine(1, samp_rate, samp_rate//SamplesPerSymbol, rolloff, 2048), 1)
        self.digital_linear_equalizer_0 = digital.linear_equalizer(numTaps, 1, AdaptiveEqualizer, True, [ ], 'corr_est')
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=Constellation,
            differential=False,
            samples_per_symbol=SamplesPerSymbol,
            pre_diff_code=True,
            excess_bw=rolloff,
            verbose=False,
            log=False,
            truncate=False)
        self.blocks_vector_to_stream_0 = blocks.vector_to_stream(gr.sizeof_gr_complex*1, numTaps)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, SamplesPerSymbol)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, delay)
        self.blocks_add_xx_0_0 = blocks.add_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 256, 40000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noiseStd, 0)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.filter_fft_rrc_filter_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.qtgui_freq_sink_x_1, 2))
        self.connect((self.blocks_add_xx_0_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.qtgui_freq_sink_x_1, 0))
        self.connect((self.blocks_delay_0, 0), (self.filter_fft_rrc_filter_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.qtgui_const_sink_x_0, 1))
        self.connect((self.blocks_vector_to_stream_0, 0), (self.qtgui_time_sink_x_1, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_delay_0_0, 0))
        self.connect((self.digital_linear_equalizer_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.digital_linear_equalizer_0, 1), (self.blocks_vector_to_stream_0, 0))
        self.connect((self.filter_fft_rrc_filter_0, 0), (self.fir_filter_xxx_0, 0))
        self.connect((self.filter_fft_rrc_filter_0_0, 0), (self.qtgui_eye_sink_x_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.digital_linear_equalizer_0, 0))
        self.connect((self.fir_filter_xxx_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.qtgui_freq_sink_x_1, 1))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "simDuo")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_chBW(0.75*self.samp_rate/self.SamplesPerSymbol)
        self.channels_fading_model_0.set_fDTs(0.005/self.samp_rate)
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate//self.SamplesPerSymbol, self.rolloff, 2048))
        self.filter_fft_rrc_filter_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate//self.SamplesPerSymbol, self.rolloff, 2048))
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.chBW, 10, window.WIN_HAMMING, 6.76))
        self.qtgui_eye_sink_x_0.set_samp_rate(self.samp_rate)
        self.qtgui_freq_sink_x_1.set_frequency_range(0, self.samp_rate)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate/self.SamplesPerSymbol)

    def get_SamplesPerSymbol(self):
        return self.SamplesPerSymbol

    def set_SamplesPerSymbol(self, SamplesPerSymbol):
        self.SamplesPerSymbol = SamplesPerSymbol
        self.set_chBW(0.75*self.samp_rate/self.SamplesPerSymbol)
        self.blocks_delay_0_0.set_dly(self.SamplesPerSymbol)
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate//self.SamplesPerSymbol, self.rolloff, 2048))
        self.filter_fft_rrc_filter_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate//self.SamplesPerSymbol, self.rolloff, 2048))
        self.qtgui_eye_sink_x_0.set_samp_per_symbol(self.SamplesPerSymbol)
        self.qtgui_time_sink_x_1.set_samp_rate(self.samp_rate/self.SamplesPerSymbol)

    def get_Constellation(self):
        return self.Constellation

    def set_Constellation(self, Constellation):
        self.Constellation = Constellation

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff
        self.filter_fft_rrc_filter_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate//self.SamplesPerSymbol, self.rolloff, 2048))
        self.filter_fft_rrc_filter_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, self.samp_rate//self.SamplesPerSymbol, self.rolloff, 2048))

    def get_preconv(self):
        return self.preconv

    def set_preconv(self, preconv):
        self.preconv = preconv

    def get_numTaps(self):
        return self.numTaps

    def set_numTaps(self, numTaps):
        self.numTaps = numTaps

    def get_noiseStd(self):
        return self.noiseStd

    def set_noiseStd(self, noiseStd):
        self.noiseStd = noiseStd
        self.analog_noise_source_x_0.set_amplitude(self.noiseStd)

    def get_delay(self):
        return self.delay

    def set_delay(self, delay):
        self.delay = delay
        self.blocks_delay_0.set_dly(self.delay)

    def get_chBW(self):
        return self.chBW

    def set_chBW(self, chBW):
        self.chBW = chBW
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.chBW, 10, window.WIN_HAMMING, 6.76))

    def get_AdaptiveEqualizer(self):
        return self.AdaptiveEqualizer

    def set_AdaptiveEqualizer(self, AdaptiveEqualizer):
        self.AdaptiveEqualizer = AdaptiveEqualizer




def main(top_block_cls=simDuo, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
