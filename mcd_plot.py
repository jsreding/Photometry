from ft import fft
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.optimize import leastsq
import K2plot

g = open("./SDSSJ1252-0234_g_frames_2018-05-19.lc1", 'r')
i = open("../i_frames/SDSSJ1252-0234_i_frames_2018-05-19.lc1", 'r')
lines_g = g.readlines()
lines_i = i.readlines()
objname = "SDSSJ1252-0234"
time_g = np.zeros(len(lines_g))
time_i = np.zeros(len(lines_i))
flux_g = np.zeros(len(lines_g))
flux_i = np.zeros(len(lines_i))
err_g = np.zeros(len(lines_g))
err_i = np.zeros(len(lines_i))
j = 0
k = 0
for l in lines_g:
    data = l.split('\t')
    time_g[j] = (float(data[1])-2458257.73477)*86400
    flux_g[j] = float(data[0])
    err_g[j] = np.sqrt(float(data[0]))
    j += 1
for l in lines_i:
    data = l.split('\t')
    time_i[k] = (float(data[1])-2458257.73492)*86400
    flux_i[k] = float(data[0])
    err_i[k] = np.sqrt(float(data[0]))
    k += 1

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

print(len(flux_g))
norm_g = flux_g/np.average(flux_g)-1
normerr_g = np.asarray(err_g)/flux_g
freq_g, per_g, fq_g, ft_g = fft(time_g, norm_g)

plt.figure()
# ax1.set_title("Kepler FFT with Nyquist ambiguities, and 2.1m FFT confirming true peak", fontsize=18)
# ax1.minorticks_on()
# ax1.set_ylabel('Rel. Flux (%)', fontsize=14)
# ax1.set_xlabel('Time (s)', fontsize=14)
# ax1.errorbar(time_g, norm_g*100, yerr=normerr_g, fmt='o', ms=2)
plt.minorticks_on()
plt.ylabel('Normalized Amplitude', fontsize=20)
plt.xlabel(r'Frequency ($\mu Hz$)', fontsize=20)
plt.xlim([0, 4000])
plt.ylim([0, 1])
plt.plot(freq*1e6, ft*5, color="Black")
# plt.annotate(0.65, 0.9,r'Highest Peak: $%s sec (%s ppt), %s \mu Hz$'%(np.round(per_g*86400, 3), np.round(ft_g[35:].max()*100, 3), np.round(fq_g*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
# ax2.scatter(fq_g*1e6, ft_g.max()*100, marker="*", s=100, color='Green')
plt.plot(freq_g*1e6, ft_g*2, color="Red")
plt.show()

norm_i = flux_i/np.average(flux_i)-1
normerr_i = np.asarray(err_i)/flux_i
freq_i, per_i, fq_i, ft_i = fft(time_i, norm_i)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_title("SDSSJ1252-0234 in $i$", fontsize=20)
ax1.minorticks_on()
ax1.set_ylabel('Rel. Flux (%)', fontsize=14)
ax1.set_xlabel('Time (s)', fontsize=14)
ax1.errorbar(time_i, norm_i*100, yerr=normerr_i, fmt='o', ms=2)
ax2 = fig.add_subplot(212)
ax2.minorticks_on()
ax2.text(0.65, 0.9,r'Highest Peak: $%s sec (%s ppt), %s \mu Hz$'%(np.round(per_i*86400, 3), np.round(ft_i[35:].max()*100, 3), np.round(fq_i*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
ax2.set_ylabel('Amplitude (ppt)', fontsize=14)
ax2.set_xlabel(r'Possible frequencies ($\mu Hz$)', fontsize=14)
ax2.scatter(fq_i*1e6, ft_i.max()*100, marker="*", s=100, color='Green')
ax2.set_xlim([0, 10000])
ax2.plot(freq_i*1e6, ft_i*100)
plt.show()

# optimize_func_g = lambda x_g: x_g[0]*np.sin(time_g/317.2782943992043+x_g[1]) - flux_g
# est_amp_g, est_phase_g = leastsq(optimize_func_g, [5.0, 0])[0]
# fine_t = np.arange(0,max(time_g),0.1)
# data_fit_g=est_amp_g*np.sin((fine_t/317.2782943992043)%2+est_phase_g)

phase_g = (time_g/(317.2782943992043))%2
phase_i = (time_i/(317.2782943992043))%2
bins_g = np.linspace(np.nanmin(phase_g), np.nanmax(phase_g), 50)
bins_i = np.linspace(np.nanmin(phase_i), np.nanmax(phase_i), 50)
digitized_g = np.digitize(phase_g, bins_g)
digitized_i = np.digitize(phase_i, bins_i)
norm_bin_g = np.array([norm_g[digitized_g == i].mean() for i in range(1,len(bins_g))])
norm_bin_i = np.array([norm_i[digitized_i == i].mean() for i in range(1,len(bins_i))])

plt.figure()
# plt.title("Phase-folded light curves for SDSSJ1252-0234 by filter", fontsize=18)
plt.xlabel("Phase", fontsize=20)
plt.ylabel("Rel. Flux (%)", fontsize=20)
plt.scatter(bins_g[:-1], norm_bin_g*100, marker='o', label='$g$')
plt.scatter(bins_i[:-1], norm_bin_i*100, marker='s', label='$i$')
# plt.plot(fine_t, data_fit_g)
plt.xlim(0, 2)
plt.legend(fontsize=14)
plt.show()
