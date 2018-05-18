def fft(time, flux, kepler=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import lombscargle

    hm = np.array([n*47.2042e-6 for n in np.arange(7)])
    fr = np.linspace(2*np.pi*1e-8, 2*np.pi*310.e-6, 10000)
    f = abs(lombscargle(time, flux, fr))
    f = np.ma.masked_where(fr/(2*np.pi) <= 1.1574074074074074e-6, f)
    if kepler == True:
        for h in hm:
            f = np.ma.masked_where((h-0.25e-6 < fr/(2*np.pi)) & (fr/(2*np.pi) < h+0.25e-6), f)
    mfq = fr[f.argmax()]/(2*np.pi)
    p = 1/mfq/86400.
    return fr, p, mfq, f, hm
