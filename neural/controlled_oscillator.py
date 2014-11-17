"""Controlled Oscillator

This is a neural model where one group of neurons produces an oscillation
and another group of neurons controls the rate of the oscillation.

Requires Nengo, available from https://github.com/nengo/nengo
"""

import numpy as np

import nengo
from nengo.utils.functions import piecewise

# Parameters
synapse = 0.1   # time constant of the recurrent connection
f_max = 2       # maximum frequency of oscillation
T = 2           # time to hold each input for
dt = 0.001      # time step of the simulation
ideal_freqs = np.array([2, 1, 0, -1, -2])   # control signal
n_neurons = 500 # number of oscillatory neurons


# Build the neural model
model = nengo.Network()
with model:
    state = nengo.Ensemble(n_neurons=n_neurons, dimensions=3, radius=1.7)

    def feedback(x):
        x0, x1, f = x
        w = f * f_max * 2 * np.pi
        return x0 + w * synapse * x1, x1 - w * synapse * x0
    nengo.Connection(state, state[:2], function=feedback, synapse=synapse)

    freq = nengo.Ensemble(n_neurons=100, dimensions=1, radius=f_max)
    nengo.Connection(freq, state[2], synapse=synapse, transform=1.0/f_max)

    stim = nengo.Node(lambda t: 1 if t < 0.08 else 0)
    nengo.Connection(stim, state[0])

    control = piecewise({i * T: s for i, s in enumerate(ideal_freqs)})
    freq_control = nengo.Node(control)

    nengo.Connection(freq_control, freq)

    p_state = nengo.Probe(state, synapse=0.03)
    p_freq = nengo.Probe(freq, synapse=0.03)

# run the simulation
sim = nengo.Simulator(model, dt=dt)
sim.run(len(ideal_freqs) * T)

# Now extract data from the simulation and compute its overall score.
# The score for each input signal is determined by taking the fourier
# transform of the output and comparing it to the ideal fourier transform
# for a perfect sine wave of the desired frequency.  Overall score is the
# average across each input.

steps = int(T / dt)
freqs = np.fft.fftfreq(steps, d=dt)

# compute fft for each input
data = sim.data[p_state][:, 1]
data.shape = len(ideal_freqs), steps
fft = np.fft.fft(data, axis=1)

# compute ideal fft for each input
ideal_data = np.zeros_like(data)
for i, f in enumerate(ideal_freqs):
    ideal_data[i] = np.cos(2 * np.pi * f * np.arange(steps) * dt)
ideal_fft = np.fft.fft(ideal_data, axis=1)

# only consider the magnitude
fft = np.abs(fft)
ideal_fft = np.abs(ideal_fft)

# compute the normalized dot product between the actual and ideal ffts
score = np.zeros_like(ideal_freqs).astype(float)
for i in range(len(ideal_freqs)):
    score[i] = np.dot(fft[i] / np.linalg.norm(fft[i]),
                      ideal_fft[i] / np.linalg.norm(ideal_fft[i]))

print 'score:', np.mean(score)


import pylab
pylab.subplot(2, 1, 1)
lines = pylab.plot(np.fft.fftshift(freqs), np.fft.fftshift(fft, axes=1).T)
pylab.xlim(-f_max * 2, f_max * 2)
pylab.xlabel('FFT of decoded value (Hz)')
pylab.title('Score: %1.4f' % np.mean(score))
pylab.legend(lines, ['%1.3f' % s for s in score],
             loc='best', prop={'size': 8})

pylab.subplot(2, 1, 2)
lines = pylab.plot(np.arange(steps) * dt, data.T)
pylab.xlabel('decoded value')
pylab.legend(lines, ['%gHz' % f for f in ideal_freqs],
             loc='best', prop={'size': 8})

pylab.show()
