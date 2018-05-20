import numpy as np
import sys
import pyfits
import glob
import os
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.signal import lombscargle
from ft import fft

def medcomb(files):
	arr = []
	for f in files:
		arr.append(pyfits.open(f)[0].data)
	arr = np.median(arr, axis=0)
	return arr

def phot(R1, R2, objx, objy):
	y, x = np.ogrid[:imred.shape[0],:imred.shape[1]]

	ap = (y-objy)**2.0 + (x-objx)**2 <= R1**2
	raw = imred[ap]
	ann = ((y-objy)**2.0 + (x-objx)**2 <= R2**2) - ap
	bg = np.median(imred[ann])
	sub = [i - bg for i in raw]
	return np.sum(sub)

path = os.getcwd() + '/'

#read in biases, median combine
os.chdir("../bias")
biases = glob.glob("bias*.fits")
medbias = medcomb(biases)
os.chdir(path)

#Get lists of filenames by filter
lists = []
numfilts = int(raw_input("How many filter sets? "))
for n in range(numfilts):
	lists.append(raw_input("Image list %s: "%(n+1)))
for l in lists:
	try:
		filenames  = np.loadtxt(path + l, dtype=str)
	except IOError:
		print('Could not find a working list with name "%s"' %file_list)
		print('\nProgram TERMINATED\n')
		sys.exit(1)
	num_files = len(filenames)

	#Find appropriate darks, median combine
	exptime = int(pyfits.open(filenames[0])[0].header['EXPTIME'])
	os.chdir("../dark")
	darks = glob.glob("dark_"+str(exptime)+"*.fits")
	meddark = medcomb(darks)
	os.chdir(path)

	#Find appropriate flats, median combine
	flatfilt = pyfits.open(filenames[0])[0].header['FILTER']
	flatfilt = flatfilt.replace(" ", "")
	if flatfilt == "BG40":
		flatfilt = flatfilt.replace("BG", "bg")
	os.chdir("../dome_flat")
	flats = glob.glob("dome_flat_"+flatfilt+"*.fits")
	medflat = medcomb(flats)
	os.chdir(path)

	#Find target and comparison centers
	sources = np.loadtxt('phot_coords.orig', dtype=float)
	target = (int(sources[0][0]), int(sources[0][1]))
	comps = []
	for s in range(1, len(sources)):
		comp = sources[s]
		comps.append((int(comp[0]), int(comp[1])))

	targ_flux = []
	comp_flux = []
	time = []
	for f in filenames:
		im = pyfits.open(f)[0]
		imraw = im.data
		time.append(Time(im.header['DATE-OBS']+"T"+im.header['TIME-OBS'], format='isot', scale='utc').jd)
		imred = (((imraw - medbias) - (meddark - medbias))/((medflat - medbias)/np.average(medflat-medbias)))[0]

		# hdu = pyfits.PrimaryHDU(imred)
		# hdu.writeto(path+"/reduced/"+f+"_reduced.fits", clobber=True)

		tarx, tary = np.where(imred == imred[target[1]-5:target[1]+5, target[0]-5:target[0]+5].max())[1][0], np.where(imred == imred[target[1]-5:target[1]+5, target[0]-5:target[0]+5].max())[0][0]
		targ_flux.append(float(phot(5, 8, tarx, tary)))

		for c in comps:
			compx, compy = (np.where(imred == imred[c[1]-5:c[1]+5, c[0]-5:c[0]+5].max())[1][0], np.where(imred == imred[c[1]-5:c[1]+5, c[0]-5:c[0]+5].max())[0][0])
			comp_flux.append(float(phot(5, 8, compx, compy)))

	final = np.array(targ_flux)/np.array(comp_flux)*np.median(np.array(comp_flux))
	norm = ((final/np.average(final))-1)*100
	print norm
	time = (np.array(time)-time[0])

	freq, per, fq, ft = fft(time, norm)

	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax1.minorticks_on()
	ax1.set_ylabel('Rel. Flux (%)', fontsize=14)
	ax1.set_xlabel('Time (s)', fontsize=14)
	ax1.scatter(time, norm, s=2)
	ax2 = fig.add_subplot(312)
	ax2.minorticks_on()
	ax2.text(0.65, 0.9,r'Highest Peak: $%s days (%s ppt), %s \mu Hz$'%(np.round(per, 3), np.round(ft[35:].max()*20, 3), np.round(fq*1e6, 3)), horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)
	ax2.set_ylabel('Amplitude (ppt)', fontsize=14)
	ax2.set_xlabel(r'Possible frequencies ($\mu Hz$)', fontsize=14)
	ax2.set_xlim([0, 310])
	ax2.scatter(fq*1e6, ft.max()*20, marker="*", s=100, color='Green')
	ax2.plot(freq*1e6/(2*np.pi), ft*20)
	ax2.axvspan(0, 1.25, alpha=0.5, color='red')
	ax3 = fig.add_subplot(313)
	ax3.minorticks_on()
	ax3.set_ylabel('Amplitude (ppt)', fontsize=14)
	ax3.set_xlabel(r'Possible frequencies ($\mu Hz$)', fontsize=14)
	if fq*1e6 < 25:
	    ax3.set_xlim([0, 50])
	else:
	    ax3.set_xlim([fq*1e6-25, fq*1e6+25])
	ax3.scatter(fq*1e6, ft.max()*20, marker="*", s=100, color='Green')
	ax3.plot(freq*1e6/(2*np.pi), ft*20)
	ax3.axvspan(0, 1.25, alpha=0.5, color='red')
	plt.show()
