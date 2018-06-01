def fft(time, flux, kepler=False):
    import matplotlib.pyplot as plt
    import numpy as np
    # from scipy.signal import lombscargle
    from astropy.stats import LombScargle

    fr, f = LombScargle(time, flux, normalization='standard').autopower(maximum_frequency=100e-4)
    f = np.ma.masked_where(fr <= 1.1574074074074074e-6, f)
    if kepler == True:
        hm = np.array([n*47.2042e-6 for n in np.arange(7)])
        for h in hm:
            f = np.ma.masked_where((h-0.25e-6 < fr/(2*np.pi)) & (fr/(2*np.pi) < h+0.25e-6), f)
        mfq = fr[f.argmax()]
        p = 1/mfq/86400.
        return fr, p, mfq, f, hm
    else:
        mfq = fr[f.argmax()]
        p = 1/mfq/86400.
        return fr, p, mfq, f
