import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lombscargle
# import nufft
import glob
from ft import fft

# f = open(sys.argv[1], "r")
files = glob.glob("/home/jsreding/Documents/UNC/Research/Data/kepler/LC/ktwo228939929.vj.lc1")
for s in files:
    f = open(s, 'r')
    lines = f.readlines()
    objname = lines[0].split(' ')[1]+lines[0].split(' ')[2]
    time = np.zeros(len(lines)-1)
    flux = np.zeros(len(lines)-1)
    if 'go' in s:
        err = np.zeros(len(lines)-1)
    i = 0
    for l in lines[1:]:
        data = l.split(' ')
        time[i] = float(data[0])
        flux[i] = float(data[1])*100
        if 'go' in s:
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
    if 'go' in s[1]:
        ax1.errorbar(time/86400., flux, yerr=err, fmt='o', linestyle="None", ms=2)
    else:
        ax1.scatter(time/86400., flux, s=2)
    ax2 = fig.add_subplot(212)
    ax2.minorticks_on()
    ax2.text(0.65, 0.9,r'Highest Peak: $%s days (%s ppt), %s \mu Hz$'%(np.round(per*86400, 3), np.round(ft[35:].max()*100, 3), np.round(fq*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
    ax2.set_ylabel('Amplitude (ppt)', fontsize=14)
    ax2.set_xlabel(r'Possible frequencies ($\mu Hz$)', fontsize=14)
    ax2.set_xlim([0, 10000])
    ax2.scatter(fq*1e6, ft.max()*100, marker="*", s=100, color='Green')
    ax2.plot(freq*1e6, ft*100)
    # ax2.axhline(y=5*np.average(ft*100), color='r', linestyle='-')
    # ax2.axvspan(0, 1.1574074074074074, alpha=0.5, color='red')
    # for h in harm:
    #     ax2.axvspan(h*1e6-.25, h*1e6+.25, alpha=0.5, color='red')
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

    print(np.average(ft*100)*5, ft[35:].max()*100)

    # nbins = 30
    # foldtimes = (time/(per*86400.))%1
    # width = 1.0/float(nbins)
    # bins = np.zeros(nbins)
    # weights = np.zeros(nbins)
    # for i in range(len(flux)):
    #     n = int(foldtimes[i] / width)
    #     if 'go' in s:
    #         weight = err[i]**-2.0
    #         bins[n] += flux[i] * weight
    #         weights[n] += weight
    #     else:
    #         bins[n] += flux[i]
    #
    # binEdges = np.arange(nbins) * width
    # x = np.linspace(0, per, 100)
    # plt.figure()
    # plt.title("Folded Light Curve for %s"%(objname), fontsize=20)
    # plt.xlabel("Phase", fontsize=14)
    # plt.ylabel("Normalized Flux (%)", fontsize = 14)
    # plt.scatter(binEdges,bins/(len(flux)/nbins),marker='o')
    # plt.ylim(-17, 17)
    # plt.plot(x/per, (bins/(len(flux)/nbins)).max()*np.sin(2*np.pi*x/per))
    # plt.show()
