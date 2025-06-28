# Wireless Communication Operations

This document lists basic signal processing algorithms that are often
used in wireless systems. These operations are added to the RAIF runtime
so that models can integrate classical DSP steps alongside neural
network layers.

## Fast Fourier Transform (FFT)

The FFT is required for OFDM modulation and channel estimation. RAIF
provides a reference radix-2 implementation that can perform both
forward and inverse transforms.

## Finite Impulse Response (FIR) Filter

FIR filters appear in channel filtering and pulse shaping stages. A
simple reference implementation is included which computes the convolution
of an input sequence with a set of coefficients.
