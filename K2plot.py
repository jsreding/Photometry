import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import leastsq
from astropy.stats import sigma_clip
# import nufft
import glob
from ft import fft

from matplotlib import rc, rcParams
rcParams['font.family'] = 'afm'
rcParams['font.sans-serif'] = ['Helvetica']
rc('text', usetex=True)

# f = open(sys.argv[1], "r")
files = sorted(glob.glob("/home/jsreding/Documents/UNC/Research/Data/kepler/Spotted/lc1/*228939929*lc1"))
for s in range(len(files)):
    f = open(files[s], 'r')
    lines = f.readlines()
    objname = files[s].split('/')[-1].split('.')[0]
    time = np.zeros(len(lines)-1)
    flux = np.zeros(len(lines)-1)
    if 'go' in files[s]:
        err = np.zeros(len(lines)-1)
    i = 0
    for l in lines[1:]:
        data = l.split(' ')
        time[i] = float(data[0])
        flux[i] = float(data[1])*1000
        if 'go' in files[s]:
            err[i] = float(data[2])
        i += 1

    freq, per, fq, ft, harm = fft(time, flux, kepler=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.minorticks_on()
    ax1.set_title(objname, fontsize=20)
    ax1.set_ylabel('Rel. Flux (%)', fontsize=14)
    ax1.set_xlabel('Time (days)', fontsize=14)
    ax1.set_xlim([0, 80])
    if 'go' in files[s][1]:
        ax1.errorbar(time/86400., flux, yerr=err, fmt='o', linestyle="None", ms=2)
    else:
        ax1.scatter(time/86400., flux, s=2)
    ax2 = fig.add_subplot(212)
    ax2.minorticks_on()
    # ax2.text(0.65, 0.9,r'Highest Peak: $%s d (%s ppt), %s \mu Hz$'%(np.round(per*86400, 3), np.round(ft.max(), 3), np.round(fq*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
    ax2.set_title('EPIC228939929 Periodogram', fontsize=18)
    ax2.set_ylabel(r'Amplitude (ppt)', fontsize=20)
    ax2.set_xlabel(r'Possible frequencies ($\mu Hz$)', fontsize=20)
    ax2.set_xlim([0, 10000])
    # ax2.set_ylim(bottom=0)
    # ax2.scatter(fq*1e6, ft.max(), marker="*", s=100, color='Green')
    ax2.plot(freq*1e6, ft)
    ax2.axhline(y=5*np.average(ft), color='r', linestyle='-')
    ax2.axvspan(0, 1.1574074074074074, alpha=0.5, color='red')
    for h in harm:
        ax2.axvspan(h*1e6-.25, h*1e6+.25, alpha=0.5, color='red')
    # ax3 = fig.add_subplot(313)
    # ax3.minorticks_on()
    # ax3.set_ylabel('Amplitude (ppt)', fontsize=14)
    # ax3.set_xlabel(r'Possible frequencies ($\mu Hz$)', fontsize=14)
    # if fq*1e6 < 25:
    #     ax3.set_xlim([0, 10000])
    # else:
    #     ax3.set_xlim([fq*1e6-25, fq*1e6+25])
    # ax3.scatter(fq*1e6, ft.max()*100, marker="*", s=100, color='Green')
    # ax3.plot(freq*1e6, ft*100)
    # ax3.axhline(y=5*np.average(ft*100), color='r', linestyle='-')
    # ax3.axvspan(0, 1.1574074074074074, alpha=0.5, color='red')
    # for h in harm:
    #     ax3.axvspan(h*1e6-.25, h*1e6+.25, alpha=0.5, color='red')
    plt.show()

    per = per*86400

    fitfunc = lambda p, x: p[0]*np.cos(2*np.pi/p[1]*x+p[2]) + p[3]*x

    def fit_leastsq(datax, datay, per, func):
        errfunc = lambda p, x, y: func(p, x) - y
        p0 = [0.5*(datay.max()-datay.min()), per, 0., np.median(datay)]

        pfit, pcov, infodict, errmsg, success = \
            leastsq(errfunc, p0, args=(datax, datay), \
                              full_output=1)

        if (len(datay) > len(p0)) and pcov is not None:
            s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
            pcov = pcov * s_sq
        else:
            pcov = np.inf

        error = []
        for i in range(len(pfit)):
            try:
              error.append(np.absolute(pcov[i][i])**0.5)
            except:
              error.append( 0.00 )
        pfit_leastsq = pfit
        perr_leastsq = np.array(error)
        return pfit_leastsq, perr_leastsq

    pfit, perr = fit_leastsq(time, flux, per, fitfunc)
    print("\n# Fit parameters and parameter errors from lestsq method :")
    print("pfit = ", pfit, "perr = ", perr)

    stime = np.linspace(time.min(), time.max(), 100000)

    plt.figure()
    plt.plot(time, flux, 'bo', stime, fitfunc(pfit, stime), 'b-')
    plt.show()

    per = pfit[1]
    inc = 'sec'

    # numcyc = int(time[-1]/(86400))
    # begcyc = int(time[0]/(86400))
    # print(per, numcyc, begcyc)
    # amps = np.zeros(numcyc-begcyc)
    # for c in range(begcyc, numcyc):
    #     trange = np.where(np.logical_and(time>=c*(86400), time<=(c+1)*(86400)))
    #     if len(trange[0]) == 0:
    #         continue
    #     pc, successc = leastsq(errfunc, p1[:], args=(time[trange], flux[trange]))
    #     amps[c-begcyc] = abs(pc[0])
    #
    # # amps = sigma_clip(amps, sigma=5)
    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.set_title("Amplitude Stability for %s"%(objname), fontsize=20)
    # ax1.set_ylabel('Flux (%)', fontsize=18)
    # ax1.scatter(time/86400, flux, s=2, zorder=0)
    # ax1.plot(stime/86400, fitfunc(p1, stime), color='k', zorder=1)
    # ax2 = fig.add_subplot(212)
    # ax2.set_ylabel('Amplitude (%)', fontsize=18)
    # ax2.set_xlabel('1-Day Cycles', fontsize=18)
    # ax2.scatter(range(numcyc-begcyc), amps)
    # plt.show()
    per = per*2.
    flux = flux*18.15

    print(per, inc)
    phase = (time/(per))%1
    bins = np.linspace(np.nanmin(phase), np.nanmax(phase), 50)
    digitized = np.digitize(phase, bins)
    norm_bin = np.array([flux[digitized == i].mean() for i in range(1,len(bins))])

    pers = per

    if per >= 600. and per < 3600.:
        per = per/60.
        inc = 'min'
    elif per >= 3600 and per < 86400.:
        per = per/3600.
        inc = 'hour'
    elif per >= 86400:
        per = per/86400
        inc = 'day'

    plt.figure()
    plt.title(r"ktwo220257559", fontsize=18)
    plt.xlabel(r"Phase", fontsize=20)
    plt.ylabel(r"Rel. Flux ($\%$)", fontsize=20)
    plt.scatter(bins[:-1], norm_bin, s=10, marker='o', label='per = %s %s'%(round(per, 4), inc))
    plt.scatter(bins[:-1]+1, norm_bin, s=10, marker='o', color='C0')
    plt.scatter(bins[:-1]+2, norm_bin, s=10, marker='o', color='C0')
    plt.plot(stime/per, fitfunc(pfit, stime))
    # plt.plot(stime/pers, fitfunc(p1, stime))
    # plt.plot(bins[:-1], out.best_fit[:199], color='C0')
    # plt.plot(fine_t, data_fit)
    plt.xlim(0, 2)
    plt.legend(fontsize=14)
    plt.show()
