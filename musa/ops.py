import numpy as np


def linear_interpolation(tbounds, fbounds):
    """Linear interpolation between the specified bounds"""
    interp = []
    for t in range(tbounds[0], tbounds[1]):
        interp.append(fbounds[0] + (t - tbounds[0]) * ((fbounds[1] - fbounds[0]) /
                                                       (tbounds[1] - tbounds[0])))
    return interp


def interpolation(signal, unvoiced_symbol):
    tbound = [None, None]
    fbound = [None, None]
    signal_t_1 = signal[0]
    isignal = np.copy(signal)
    uv = np.ones(signal.shape, dtype=np.int8)
    for t in range(1, signal.shape[0]):
        if (signal[t] > unvoiced_symbol) and (signal_t_1 <= unvoiced_symbol) and (tbound == [None, None]):
            # First part of signal is unvoiced, set to constant first voiced
            isignal[:t] = signal[t]
            uv[:t] = 0
        elif (signal[t] <= unvoiced_symbol) and (signal_t_1 > unvoiced_symbol):
            tbound[0] = t - 1
            fbound[0] = signal_t_1
        elif (signal[t] > unvoiced_symbol) and (signal_t_1 <= unvoiced_symbol):
            tbound[1] = t
            fbound[1] = signal[t]
            isignal[tbound[0]:tbound[1]] = linear_interpolation(tbound, fbound)
            uv[tbound[0]:tbound[1]] = 0
            # reset values
            tbound = [None, None]
            fbound = [None, None]
        signal_t_1 = signal[t]
    # now end of signal if necessary
    if tbound[0] is not None:
        isignal[tbound[0]:] = fbound[0]
        uv[tbound[0]:] = 0
    return isignal, uv
