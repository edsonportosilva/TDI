#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Transmissão digital num canal AWGN
# Author: Edson Porto da Silva
# Copyright: 2023
# GNU Radio version: 3.10.7.0

from packaging.version import Version as StrictVersion
from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import sip



class top_block(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Transmissão digital num canal AWGN", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Transmissão digital num canal AWGN")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
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

        self.settings = Qt.QSettings("GNU Radio", "top_block")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 5120000
        self.sps = sps = 8
        self.gain = gain = 0
        self.fc = fc = samp_rate/4
        self.constellation = constellation = digital.constellation_16qam().base()
        self.RRCrolloff = RRCrolloff = 0.01
        self.Pn = Pn = -50

        ##################################################
        # Blocks
        ##################################################

        self._gain_range = Range(-20, 20, 0.1, 0, 400)
        self._gain_win = RangeWidget(self._gain_range, self.set_gain, "Ganho no sinal (dB)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_win)
        self._fc_range = Range(samp_rate/16, samp_rate/2, 10, samp_rate/4, 400)
        self._fc_win = RangeWidget(self._fc_range, self.set_fc, "Frequencia da portadora (Hz)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._fc_win)
        self._Pn_range = Range(-100, 0, 0.1, -50, 400)
        self._Pn_win = RangeWidget(self._Pn_range, self.set_Pn, "Potência de ruído (dBm)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._Pn_win, 3, 0, 1, 2)
        for r in range(3, 4):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.root_raised_cosine_filter_0_0_1_1 = filter.fir_filter_ccf(
            sps,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                (samp_rate/sps),
                RRCrolloff,
                4001))
        self.root_raised_cosine_filter_0_0_1_0_1 = filter.fir_filter_fff(
            sps,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                (samp_rate/sps),
                RRCrolloff,
                4001))
        self.root_raised_cosine_filter_0_0_1_0_0 = filter.fir_filter_ccf(
            1,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                (samp_rate/sps),
                RRCrolloff,
                4001))
        self.root_raised_cosine_filter_0_0_1_0 = filter.fir_filter_ccf(
            sps,
            firdes.root_raised_cosine(
                1,
                samp_rate,
                (samp_rate/sps),
                RRCrolloff,
                4001))
        self.qtgui_number_sink_1 = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_NONE,
            1,
            None # parent
        )
        self.qtgui_number_sink_1.set_update_time(0.10)
        self.qtgui_number_sink_1.set_title('Taxa de erro de bit [Bit error rate (BER)]')

        labels = ['BER', '', '', '', '',
            '', '', '', '', '']
        units = ['', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(1):
            self.qtgui_number_sink_1.set_min(i, 0)
            self.qtgui_number_sink_1.set_max(i, 0.5)
            self.qtgui_number_sink_1.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_1.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_1.set_label(i, labels[i])
            self.qtgui_number_sink_1.set_unit(i, units[i])
            self.qtgui_number_sink_1.set_factor(i, factor[i])

        self.qtgui_number_sink_1.enable_autoscale(False)
        self._qtgui_number_sink_1_win = sip.wrapinstance(self.qtgui_number_sink_1.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_number_sink_1_win, 1, 1, 1, 1)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_number_sink_0 = qtgui.number_sink(
            gr.sizeof_float,
            0.1,
            qtgui.NUM_GRAPH_NONE,
            3,
            None # parent
        )
        self.qtgui_number_sink_0.set_update_time(0.10)
        self.qtgui_number_sink_0.set_title('')

        labels = ['SNR por bit [Eb/N0] ', 'Nível de sinal (dBm)', 'Nível de ruído (dBm)', '', '',
            '', '', '', '', '']
        units = ['dB', '', '', '', '',
            '', '', '', '', '']
        colors = [("black", "red"), ("blue", "red"), ("black", "black"), ("black", "black"), ("black", "black"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(3):
            self.qtgui_number_sink_0.set_min(i, -1)
            self.qtgui_number_sink_0.set_max(i, 1)
            self.qtgui_number_sink_0.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_0.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_0.set_label(i, labels[i])
            self.qtgui_number_sink_0.set_unit(i, units[i])
            self.qtgui_number_sink_0.set_factor(i, factor[i])

        self.qtgui_number_sink_0.enable_autoscale(False)
        self._qtgui_number_sink_0_win = sip.wrapinstance(self.qtgui_number_sink_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_number_sink_0_win, 2, 1, 1, 1)
        for r in range(2, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_f(
            4096, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            'Espectro do sinal modulado', #name
            2,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-120), (-20))
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(0.2)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)


        self.qtgui_freq_sink_x_0.set_plot_pos_half(not False)

        labels = ['Sinal + Ruido', 'Sinal ', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_freq_sink_x_0_win, 1, 0, 2, 1)
        for r in range(1, 3):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_eye_sink_x_0_0 = qtgui.eye_sink_c(
            (1024*8), #size
            samp_rate, #samp_rate
            1, #number of inputs
            None
        )
        self.qtgui_eye_sink_x_0_0.set_update_time(0.10)
        self.qtgui_eye_sink_x_0_0.set_samp_per_symbol(sps)
        self.qtgui_eye_sink_x_0_0.set_y_axis(-2, 2)

        self.qtgui_eye_sink_x_0_0.set_y_label('Diagrama de olho', "")

        self.qtgui_eye_sink_x_0_0.enable_tags(False)
        self.qtgui_eye_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_eye_sink_x_0_0.enable_autoscale(False)
        self.qtgui_eye_sink_x_0_0.enable_grid(True)
        self.qtgui_eye_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_eye_sink_x_0_0.enable_control_panel(True)


        labels = ['Real', 'Imag', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'blue', 'blue', 'blue',
            'blue', 'blue', 'blue', 'blue', 'blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_eye_sink_x_0_0.set_line_label(i, "Eye [Re{{Data {0}}}]".format(round(i/2)))
                else:
                    self.qtgui_eye_sink_x_0_0.set_line_label(i, "Eye [Im{{Data {0}}}]".format(round((i-1)/2)))
            else:
                self.qtgui_eye_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_eye_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_eye_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_eye_sink_x_0_0.set_line_style(i, styles[i])
            self.qtgui_eye_sink_x_0_0.set_line_marker(i, markers[i])
            self.qtgui_eye_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_eye_sink_x_0_0_win = sip.wrapinstance(self.qtgui_eye_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_eye_sink_x_0_0_win, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=constellation,
            differential=True,
            samples_per_symbol=sps,
            pre_diff_code=True,
            excess_bw=RRCrolloff,
            verbose=False,
            log=False,
            truncate=False)
        self.digital_constellation_decoder_cb_0_0 = digital.constellation_decoder_cb(constellation)
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(constellation)
        self.blocks_xor_xx_0 = blocks.xor_bb()
        self.blocks_transcendental_0 = blocks.transcendental('tanh', "float")
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, samp_rate,True)
        self.blocks_sub_xx_0_0_0 = blocks.sub_ff(1)
        self.blocks_rms_xx_0_0_0 = blocks.rms_cf(0.00001)
        self.blocks_rms_xx_0_0 = blocks.rms_ff(0.00001)
        self.blocks_nlog10_ff_0_1_0 = blocks.nlog10_ff(20, 1, (-24))
        self.blocks_nlog10_ff_0_1 = blocks.nlog10_ff(20, 1, (-30))
        self.blocks_multiply_xx_1_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0_1_0 = blocks.multiply_vff(1)
        self.blocks_multiply_xx_0_1 = blocks.multiply_vff(1)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vff(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vff(1)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_cc(numpy.sqrt(10**(gain/10)))
        self.blocks_moving_average_xx_1 = blocks.moving_average_cc(1000, 1/1000, 4000, 1)
        self.blocks_moving_average_xx_0_1 = blocks.moving_average_ff(1000000, (1/(8*1000000)), 50000, 1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_divide_xx_0 = blocks.divide_cc(1)
        self.blocks_complex_to_real_0_0_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_real_0_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_real_0 = blocks.complex_to_real(1)
        self.blocks_complex_to_imag_0_1 = blocks.complex_to_imag(1)
        self.blocks_complex_to_imag_0_0 = blocks.complex_to_imag(1)
        self.blocks_complex_to_imag_0 = blocks.complex_to_imag(1)
        self.blocks_char_to_float_0 = blocks.char_to_float(1, 1)
        self.blocks_add_xx_0_0_0 = blocks.add_vff(1)
        self.blocks_add_xx_0_0 = blocks.add_vff(1)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, fc, 1, 0, 0)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, fc, 1, 0, 0)
        self.analog_random_source_x_1 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 256, 10000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_f(analog.GR_GAUSSIAN, (numpy.sqrt(sps*10**(Pn/10)*1e-3)), 0)
        self.analog_const_source_x_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, (10*numpy.log10(numpy.log2(constellation.arity()))))
        self.analog_agc_xx_0_0 = analog.agc_cc((1e-4), 1, 1.0, 65536)
        self.analog_agc_xx_0 = analog.agc_cc((1e-4), 1, 1.0, 65536)
        self.Constelacoes = qtgui.const_sink_c(
            4096, #size
            'Diagrama de constelações', #name
            2, #number of inputs
            None # parent
        )
        self.Constelacoes.set_update_time(0.05)
        self.Constelacoes.set_y_axis((-1.5), 1.5)
        self.Constelacoes.set_x_axis((-1.5), 1.5)
        self.Constelacoes.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.Constelacoes.enable_autoscale(False)
        self.Constelacoes.enable_grid(True)
        self.Constelacoes.enable_axis_labels(True)


        labels = ['Recebida', 'Transmitida', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(2):
            if len(labels[i]) == 0:
                self.Constelacoes.set_line_label(i, "Data {0}".format(i))
            else:
                self.Constelacoes.set_line_label(i, labels[i])
            self.Constelacoes.set_line_width(i, widths[i])
            self.Constelacoes.set_line_color(i, colors[i])
            self.Constelacoes.set_line_style(i, styles[i])
            self.Constelacoes.set_line_marker(i, markers[i])
            self.Constelacoes.set_line_alpha(i, alphas[i])

        self._Constelacoes_win = sip.wrapinstance(self.Constelacoes.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._Constelacoes_win, 0, 1, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 2):
            self.top_grid_layout.setColumnStretch(c, 1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.analog_agc_xx_0, 0), (self.blocks_multiply_xx_1_0, 0))
        self.connect((self.analog_agc_xx_0_0, 0), (self.Constelacoes, 1))
        self.connect((self.analog_agc_xx_0_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.analog_agc_xx_0_0, 0), (self.digital_constellation_decoder_cb_0_0, 0))
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_sub_xx_0_0_0, 2))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0_0_0, 0))
        self.connect((self.analog_noise_source_x_0, 0), (self.root_raised_cosine_filter_0_0_1_0_1, 0))
        self.connect((self.analog_random_source_x_1, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_complex_to_imag_0, 0))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_complex_to_real_0_0, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_complex_to_imag_0_1, 0))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_complex_to_real_0_0_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.blocks_add_xx_0_0_0, 1))
        self.connect((self.blocks_add_xx_0_0, 0), (self.qtgui_freq_sink_x_0, 1))
        self.connect((self.blocks_add_xx_0_0_0, 0), (self.blocks_multiply_xx_0_1, 0))
        self.connect((self.blocks_add_xx_0_0_0, 0), (self.blocks_multiply_xx_0_1_0, 0))
        self.connect((self.blocks_add_xx_0_0_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_char_to_float_0, 0), (self.blocks_transcendental_0, 0))
        self.connect((self.blocks_complex_to_imag_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_complex_to_imag_0_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_complex_to_imag_0_1, 0), (self.blocks_multiply_xx_0_1, 1))
        self.connect((self.blocks_complex_to_real_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_complex_to_real_0_0, 0), (self.blocks_multiply_xx_0_0, 0))
        self.connect((self.blocks_complex_to_real_0_0_0, 0), (self.blocks_multiply_xx_0_1_0, 1))
        self.connect((self.blocks_divide_xx_0, 0), (self.blocks_moving_average_xx_1, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.root_raised_cosine_filter_0_0_1_1, 0))
        self.connect((self.blocks_moving_average_xx_0_1, 0), (self.qtgui_number_sink_1, 0))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.blocks_multiply_xx_1_0, 1))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.blocks_complex_to_imag_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.blocks_complex_to_real_0, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.root_raised_cosine_filter_0_0_1_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.blocks_multiply_xx_0_1, 0), (self.blocks_float_to_complex_0, 1))
        self.connect((self.blocks_multiply_xx_0_1_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_multiply_xx_1_0, 0), (self.Constelacoes, 0))
        self.connect((self.blocks_multiply_xx_1_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.blocks_nlog10_ff_0_1, 0), (self.blocks_sub_xx_0_0_0, 0))
        self.connect((self.blocks_nlog10_ff_0_1, 0), (self.qtgui_number_sink_0, 1))
        self.connect((self.blocks_nlog10_ff_0_1_0, 0), (self.blocks_sub_xx_0_0_0, 1))
        self.connect((self.blocks_nlog10_ff_0_1_0, 0), (self.qtgui_number_sink_0, 2))
        self.connect((self.blocks_rms_xx_0_0, 0), (self.blocks_nlog10_ff_0_1_0, 0))
        self.connect((self.blocks_rms_xx_0_0_0, 0), (self.blocks_nlog10_ff_0_1, 0))
        self.connect((self.blocks_sub_xx_0_0_0, 0), (self.qtgui_number_sink_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.blocks_throttle_0, 0), (self.root_raised_cosine_filter_0_0_1_0_0, 0))
        self.connect((self.blocks_transcendental_0, 0), (self.blocks_moving_average_xx_0_1, 0))
        self.connect((self.blocks_xor_xx_0, 0), (self.blocks_char_to_float_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.blocks_xor_xx_0, 1))
        self.connect((self.digital_constellation_decoder_cb_0_0, 0), (self.blocks_xor_xx_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.root_raised_cosine_filter_0_0_1_0, 0), (self.analog_agc_xx_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0_0_1_0, 0), (self.blocks_rms_xx_0_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0_0_1_0_0, 0), (self.qtgui_eye_sink_x_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0_0_1_0_1, 0), (self.blocks_rms_xx_0_0, 0))
        self.connect((self.root_raised_cosine_filter_0_0_1_1, 0), (self.analog_agc_xx_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "top_block")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_fc(self.samp_rate/4)
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.blocks_throttle_0.set_sample_rate(self.samp_rate)
        self.qtgui_eye_sink_x_0_0.set_samp_rate(self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(0, self.samp_rate)
        self.root_raised_cosine_filter_0_0_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_0_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.analog_noise_source_x_0.set_amplitude((numpy.sqrt(self.sps*10**(self.Pn/10)*1e-3)))
        self.qtgui_eye_sink_x_0_0.set_samp_per_symbol(self.sps)
        self.root_raised_cosine_filter_0_0_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_0_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.blocks_multiply_const_vxx_1.set_k(numpy.sqrt(10**(self.gain/10)))

    def get_fc(self):
        return self.fc

    def set_fc(self, fc):
        self.fc = fc
        self.analog_sig_source_x_0.set_frequency(self.fc)
        self.analog_sig_source_x_0_0.set_frequency(self.fc)

    def get_constellation(self):
        return self.constellation

    def set_constellation(self, constellation):
        self.constellation = constellation
        self.digital_constellation_decoder_cb_0.set_constellation(self.constellation)
        self.digital_constellation_decoder_cb_0_0.set_constellation(self.constellation)

    def get_RRCrolloff(self):
        return self.RRCrolloff

    def set_RRCrolloff(self, RRCrolloff):
        self.RRCrolloff = RRCrolloff
        self.root_raised_cosine_filter_0_0_1_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_0_0.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_0_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))
        self.root_raised_cosine_filter_0_0_1_1.set_taps(firdes.root_raised_cosine(1, self.samp_rate, (self.samp_rate/self.sps), self.RRCrolloff, 4001))

    def get_Pn(self):
        return self.Pn

    def set_Pn(self, Pn):
        self.Pn = Pn
        self.analog_noise_source_x_0.set_amplitude((numpy.sqrt(self.sps*10**(self.Pn/10)*1e-3)))




def main(top_block_cls=top_block, options=None):

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
