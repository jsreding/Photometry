from ft import fft
import matplotlib.pyplot as plt
import glob
import numpy as np
from scipy.optimize import leastsq
# import K2plot

from matplotlib import rc, rcParams
rcParams['font.family'] = 'afm'
rcParams['font.sans-serif'] = ['Helvetica']
rc('text', usetex=True)

f = open("/home/jsreding/Documents/UNC/Research/Data/mcdonald/SDSSJ1252-0234/sdssj1252-0234-all_g.lc1", 'r')
i = open("/home/jsreding/Documents/UNC/Research/Data/mcdonald/SDSSJ1252-0234/sdssj1252-0234-all_i.lc1", 'r')
lines = f.readlines()
lines_i = i.readlines()
objname = "SDSSJ1252-0234"
time = np.zeros(len(lines))
time_i = np.zeros(len(lines_i))
flux = np.zeros(len(lines))
flux_i = np.zeros(len(lines_i))
# err = np.zeros(len(lines))
# err_i = np.zeros(len(lines_i))
j = 0
k = 0
for l in lines:
    if not l.startswith("#"):
        data = l.strip().split(' ')
        if data[1] != '--':
            time[j] = float(data[0])
            flux[j] = float(data[1])*100.
            # err[j] = np.sqrt(abs(float(data[1])))
            j += 1
for l in lines_i:
    if not l.startswith("#"):
        data = l.strip().split(' ')
        if data[1] != '--':
            time_i[k] = float(data[0])
            flux_i[k] = float(data[1])*100.
    #         # err_i[k] = np.sqrt(abs(float(data[0])))
            k += 1

files = glob.glob("/home/jsreding/Documents/UNC/Research/Data/kepler/Spotted/lc1/*228939929*lc1")
for s in files:
    f = open(s, 'r')
    lines = f.readlines()
    # objname = lines[0].split(' ')[1]+lines[0].split(' ')[2]
    ktime = np.zeros(len(lines)-1)
    kflux = np.zeros(len(lines)-1)
    if 'go' in s:
        err = np.zeros(len(lines)-1)
    i = 0
    for l in lines[1:]:
        data = l.split(' ')
        ktime[i] = float(data[0])
        kflux[i] = float(data[1])*100.
        if 'go' in s:
            err[i] = float(data[2])
        i += 1

kfreq, kper, kfq, kft, kharm = fft(ktime, kflux, kepler=True)

# normerr = err/flux
freq, per, fq, ft = fft(time, flux)

fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.set_title("SDSSJ1252-0234", fontsize=18)
# ax1.minorticks_on()
# ax1.set_ylabel('Rel. Flux (%)', fontsize=14)
# ax1.set_xlabel('Time (s)', fontsize=14)
# # ax1.errorbar(time, norm, yerr=err, fmt='o', ms=2)
# ax1.scatter(time, norm, s=10)
ax2 = fig.add_subplot(111)
ax2.minorticks_on()
ax2.set_title("K2/McDonald J1252 Periodograms", fontsize=24)
# ax2.text(0.5, 0.95,r'Peak: $%s sec (%s ppt), %s \mu Hz$'%(np.round(per*86400, 3), np.round(ft.max()*5, 3), np.round(fq*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
ax2.set_ylabel(r'Amplitude (\%)', fontsize=20)
ax2.set_xlabel(r'Frequency ($\mu Hz$)', fontsize=20)
plt.xlim([0, 4000])
# plt.ylim([0, 6.])
ax2.plot(kfreq*1e6, kft, color="Black", label=r'K2($\times18.15$)')
# plt.annotate(0.65, 0.9,r'Highest Peak: $%s sec (%s ppt), %s \mu Hz$'%(np.round(per*86400, 3), np.round(ft[35:].max()*100, 3), np.round(fq*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
# ax2.scatter(fq*1e6, ft.max()*5, marker="*", s=100, color='Green')
plt.plot(freq*1e6, ft, color="Red", label='McDonald')
plt.legend(fontsize=18)
plt.show()

per = 315.95764728593076

fitfunc = lambda p, x: p[0]*np.sin(2*np.pi/p[1]*x+2*np.pi*p[2]) + p[3]*x

#For fixing period: https://sites.google.com/site/theodoregoetz/notes/settingboundedandfixedparametersinscipyfittingroutines
lbound = lambda p, x: 1e4*np.sqrt(p-x) + 1e-3*(p-x) if (x<p) else 0
ubound = lambda p, x: 1e4*np.sqrt(x-p) + 1e-3*(x-p) if (x>p) else 0
bound  = lambda p, x: lbound(p[0],x) + ubound(p[1],x)
fixed  = lambda p, x: bound((p,p), x)

def fit_leastsq(datax, datay, p_in, func):
    errfunc = lambda p, x, y: func(p, x) - y

    pfit, pcov, infodict, errmsg, success = \
        leastsq(errfunc, p_in, args=(datax, datay), \
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

p0 = [np.abs(0.5*(flux.max()-flux.min())), per, 0., 0.]
p0_i = [np.abs(0.5*(flux_i.max()-flux_i.min())), per, 0., 0.]
pfit, perr = fit_leastsq(time, flux, p0, fitfunc)
pfit_i, perr_i = fit_leastsq(time_i, flux_i, p0_i, fitfunc)

print("\n# Fit parameters and parameter errors from lestsq method :")
print("pfit = ", pfit, "perr = ", perr)
print("pfit_i = ", pfit_i, "perr_i = ", perr_i)

print("i-g phase (deg):", abs(pfit[2]-pfit_i[2])/(np.pi)*180, '+/-', np.sqrt(np.degrees(perr[2])**2 + np.degrees(perr_i[2])**2))

stime = np.linspace(time.min(), time.max(), 10000000)
stime_i = np.linspace(time_i.min(), time_i.max(), 10000000)

plt.figure()
plt.plot(time, flux, 'bo', stime, fitfunc(pfit, stime), 'b-')
plt.plot(time_i, flux_i, 'ro', stime_i, fitfunc(pfit_i, stime_i), 'r-')
plt.show()
per = 315.95764728593076

pfit[0] = 4.73830561 #0.1585620315
pfit[1] = 315.95764728593076
pfit[2] = 0.246897 #0.005325939945
pfit[3] = 0.
pfit_i[0] = 2.01180725 #0.1625557520
pfit_i[1] = 315.95764728593076
pfit_i[2] = 0.293791 #0.01285985599
pfit_i[3] = 0.
inc = 'sec'

print("\n# Fit parameters and parameter errors from lestsq method :")
print("pfit = ", pfit, "perr = ", perr)
print("pfit_i = ", pfit_i, "perr_i = ", perr_i)

# print("i-g phase (deg):", abs(pfit[2]-pfit_i[2])/(np.pi)*180, '+/-', np.sqrt(np.degrees(perr[2])**2 + np.degrees(perr_i[2])**2))

print(per, inc)
phase = (time/(per))%1
bins = np.linspace(np.nanmin(phase), np.nanmax(phase), 50)
digitized = np.digitize(phase, bins)
norm_bin = np.array([flux[digitized == i].mean() for i in range(1,len(bins))])
phase_i = (time_i/(per))%1
bins_i = np.linspace(np.nanmin(phase_i), np.nanmax(phase_i), 50)
digitized_i = np.digitize(phase_i, bins_i)
norm_bin_i = np.array([flux_i[digitized_i == i].mean() for i in range(1,len(bins_i))])

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
plt.xlabel("Phase", fontsize=20)
plt.ylabel(r"Rel. Flux ($\%$)", fontsize=20)
plt.scatter(bins[:-1], norm_bin, s=30, marker='o', label='$g$ frames', color='c')
plt.scatter(bins[:-1]+1, norm_bin, s=30, marker='o', color='c')
plt.plot(stime/per, fitfunc(pfit, stime), color='c')
plt.scatter(bins_i[:-1], norm_bin_i, s=30, marker='o', color='m', label='$i$ frames')
plt.scatter(bins_i[:-1]+1, norm_bin_i, s=30, marker='o', color='m')
plt.plot(stime_i/per, fitfunc(pfit_i, stime_i), color='m')
# plt.plot(bins[:-1], out.best_fit[:199], color='C0')
# plt.plot(fine_t, data_fit)
plt.xlim(0, 2)
plt.plot([], [], ' ', label='per = %s %s'%(round(per, 2), inc))
plt.legend(fontsize=14)
plt.show()
