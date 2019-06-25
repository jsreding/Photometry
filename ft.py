def fft(time, flux, kepler=False):
    import numpy as np
    from astropy.stats import LombScargle

    fr, f = LombScargle(time, flux).autopower(minimum_frequency=1e-9, maximum_frequency=10000e-6, samples_per_peak=50, normalization='standard', method='fast')
    # f = np.ma.masked_where(fr <= 3160e-6, f)
    # f = np.ma.masked_where(fr >= 3170e-6, f)
    f = np.ma.masked_where(fr <= 1.1574074074074074e-6, f)
    if kepler == True:
        hm = np.array([n*47.2042e-6 for n in np.arange(50)])
        for h in hm:
            f = np.ma.masked_where((h-0.25e-6 <= fr) & (fr <= h+0.25e-6), f)
        mfq = fr[f.argmax()]
        # print(f.argmax())
        p = 1/mfq/86400.
        return fr, p, mfq, f, hm
    else:
        mfq = fr[f.argmax()]
        p = 1/mfq/86400.
        return fr, p, mfq, f
