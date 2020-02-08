###########################
# Photometry script for images from SOAR and McDonald Observatory
# Joshua Reding
# Last update: 2020-02-07
###########################

import numpy as np
import sys
import astropy.io.fits as fits
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime
from astropy import time, coordinates as coord, units as u
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clip, gaussian_sigma_to_fwhm
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, centroids
import csv
from ft import fft

###########################

def medcomb(files):
	arr = []
	for f in files:
		arr.append(fits.open(f)[0].data)
	arr = np.median(arr, axis=0)
	return arr

def phot(img, R1, R2, objx, objy):
	ap = CircularAperture((objx, objy), R1)
	print(ap)
	ann = CircularAnnulus((objx, objy), R1+5, R2+5)
	apers = [ap, ann]
	ph = aperture_photometry(img, apers)
	bg = float(ph['aperture_sum_1'])/ann.area*ap.area
	sub = float(ph['aperture_sum_0']) - bg
	print("NET FLUX IN APERTURE:", sub)
	print()
	return sub

def optimizephot(obj, img):
	ix, iy = obj
	cenx, ceny = centroids.centroid_2dg(img[iy-10:iy+10, ix-10:ix+10])
	cenx += ix-10
	ceny += iy-10
	print("CENTROIDED LOCATION:", cenx, ceny)

	data = np.ma.asanyarray(img)
	weights = np.ones(data.shape)
	init_con = np.median(data)
	init_amp = np.max(data[int(ceny)-5:int(ceny)+5, int(cenx)-5:int(cenx)+5])

	g_init = centroids.GaussianConst2D(constant=init_con, amplitude=init_amp, x_mean=cenx, y_mean=ceny, x_stddev=1, y_stddev=1, theta=0, fixed={'x_mean': True, 'y_mean': True})
	fitter = LevMarLSQFitter()
	y, x = np.indices(data.shape)
	gfit = fitter(g_init, x, y, data, weights=weights)
	print(gfit)

	ap = np.round(1.6*np.sqrt(gfit.x_stddev**2. + gfit.y_stddev**2), 1) #Mighell 1999ASPC..189...50M
	print("OPTIMAL APERTURE (unscaled):", ap)
	if ap > 15. or ap <= 1.:
		print("~~~~~~~~ POSSIBLE BAD APERTURE FIT - CHECK IMAGE ~~~~~~~~")
	print()
	amp = gfit.amplitude
	return cenx, ceny, ap, amp

def main():
	path = os.getcwd() + '/'

	print("### SINGLE FILTER APERTURE PHOTOMETRY ###")
	print("THIS DIRECTORY MUST CONTAIN:")
	print("FOLDERS: ./bias ./flat ./raw (./dark); FILES: ilist, phot_coords")
	print()

	if not os.path.exists(path+"bias/"):
		sys.exit("ERROR: NO bias FOLDER FOUND; ENSURE DIRECTORY CONTAINS ./bias ./flat ./raw (./dark)")
	elif not os.path.exists(path+"flat/"):
		sys.exit("ERR: NO flat FOLDER FOUND; ENSURE DIRECTORY CONTAINS ./bias ./flat ./raw (./dark)")

	site = input("OBSERVING SITE (s = SOAR, m = McDonald): ")
	if site != "m" and site != "s":
		sys.exit("ERROR: UNRECOGNIZED LOCATION")
	print()

	print("*** LOCATING IMAGE LIST ***")
	try:
		filenames  = sorted(np.loadtxt(path+"ilist", dtype=str))
	except IOError:
		sys.exit('ERROR: COULD NOT FIND ilist; GENERATE IN ./raw WITH "ls -1 > ilist" AND MOVE HERE')
	num_files = len(filenames)
	print(num_files, "IMAGES IN LIST")
	print()

	os.chdir(path+'raw/')
	im0 = fits.open(filenames[0])[0]
	fhead = im0.header

	print("*** CREATING MASTER BIAS ***")
	os.chdir(path+"bias/")
	if site == "m":
		biases = sorted(glob.glob("bias*.fits"))
	elif site == "s":
		biases = sorted(glob.glob("*zero*.fits"))
	if len(biases) == 0:
		sys.exit("ERROR: CHECK FILENAME CONVENTION")
	medbias = medcomb(biases)
	hdu = fits.PrimaryHDU(medbias)
	hdu.writeto("master_bias.fits", overwrite=True)
	print("*** master_bias.fits CREATED ***")
	print()

	if site == "m":
		exptime = int(fhead['EXPTIME'])
		os.chdir(path+"dark")
		darks = glob.glob("dark_"+str(exptime)+"*.fits")
		meddark = medcomb(darks)
		os.chdir(path)

	print("*** CREATING MASTER FLAT ***")
	os.chdir(path+"raw/")
	flatfilt = fhead['FILTER']
	if flatfilt == "<NO FILTER>":
		flatfilt = fhead['FILTER2']
	flatfilt = flatfilt.replace(" ", "")
	print("FILTER:", flatfilt)
	if flatfilt == "BG40":
		flatfilt = flatfilt.replace("BG", "bg")
	os.chdir(path+"flat/")
	if site == "m":
		flats = glob.glob("dome_flat_"+flatfilt+"*.fits")
	elif site == "s":
		flats = glob.glob("*DomeFlat*")
	if len(flats) == 0:
		sys.exit("ERROR: CHECK FILENAME CONVENTION")
	medflat = medcomb(flats)
	flatred = (medflat - medbias)/np.mean(medflat-medbias)
	hdu = fits.PrimaryHDU(flatred)
	hdu.writeto("master_flat.fits", overwrite=True)
	print("*** master_flat.fits CREATED ***")
	print()

	os.chdir(path)

	print("*** READING OBJECT AND COMPARISON COORDINATES FROM phot_coords ***")
	try:
		sources = np.loadtxt('phot_coords', dtype=float)
	except:
		sys.exit("ERROR: phot_coords NOT FOUND")
	target = (int(sources[0][0]), int(sources[0][1]))
	comps = [(int(sources[n][0]), int(sources[n][1])) for n in range(1, len(sources))]
	print("OBJECT LOCATION:", target)
	print("COMPARISON LOCATION(S):", comps)
	print()

	# Change these with new objects
	ra = "12:52:30.9336260227"
	dec = "-02:34:17.720442762"
	print("LAST OBJECT RA/DEC:", ra, dec)
	loctest = input("USE THIS SAME OBJECT ([y]/n)? ")
	if loctest == 'n':
		sys.exit("~~~~~~~~ CHANGE RA AND DEC WITHIN CODE, THEN RUN AGAIN ~~~~~~~~")
	objloc = coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
	print()

	targ_flux = np.zeros(num_files)
	comp_flux = np.zeros((len(comps), num_files))
	times = np.zeros(num_files)
	bjd = np.zeros(num_files)

	# if site == 'm':
	# 	siteloc = coord.EarthLocation.of_site('mcdonald')
	# 	firstjd = time.Time(im0.header['DATE-OBS']+"T"+im0.header['TIME-OBS'], format='isot', scale='utc', location=siteloc).jd + time.Time(im0.header['DATE-OBS']+"T"+im0.header['TIME-OBS'], format='isot', scale='utc', location=siteloc).light_travel_time(objloc).jd
	if site == 's':
		siteloc = coord.EarthLocation.of_site('gemini_south')
		print("FIRST EXPOSURE START TIME (GPS-synced UTC):", im0.header['OPENDATE']+"T"+im0.header['OPENTIME'])
		firstt = time.Time(im0.header['OPENDATE']+"T"+im0.header['OPENTIME'], format='isot', scale='utc', location=siteloc) + 0.5*(time.Time(im0.header['OPENDATE']+"T"+im0.header['CLOSETIM'], format='isot', scale='utc', location=siteloc)-time.Time(im0.header['OPENDATE']+"T"+im0.header['OPENTIME'], format='isot', scale='utc', location=siteloc)) + time.Time(im0.header['OPENDATE']+"T"+im0.header['OPENTIME'], format='isot', scale='utc', location=siteloc).light_travel_time(objloc)
		print("FIRST EXPOSURE MIDPOINT TIME (BJD_TDB):", firstt.jd)
		print()

	os.chdir(path+'raw/')

	print("DESIGNATE SUITABLE BACKGROUND REGION")
	bgxmin = int(input("xmin: "))
	bgxmax = int(input("xmax: "))
	bgymin = int(input("ymin: "))
	bgymax = int(input("ymax: "))

	for f in range(len(filenames)):
		print()
		print("############## PERFORMING REDUCTIONS ON %s ##############"%(filenames[f]))
		print()
		im = fits.open(filenames[f])[0]
		imraw = im.data
		airm = float(im.header['AIRMASS'])
		t = time.Time(im.header['OPENDATE']+"T"+im.header['OPENTIME'], format='isot', scale='utc', location=siteloc)+0.5*(time.Time(im.header['OPENDATE']+"T"+im.header['CLOSETIM'], format='isot', scale='utc', location=siteloc)-time.Time(im.header['OPENDATE']+"T"+im.header['OPENTIME'], format='isot', scale='utc', location=siteloc))
		barycorrs = time.Time(im.header['OPENDATE']+"T"+im.header['OPENTIME'], format='isot', scale='utc', location=siteloc).light_travel_time(objloc).sec
		barycorrjd = time.Time(im.header['OPENDATE']+"T"+im.header['OPENTIME'], format='isot', scale='utc', location=siteloc).light_travel_time(objloc).jd

		if site == 's':
			times[f] = round((t - firstt).sec+barycorrs, 8)
			bjd[f] = t.jd + barycorrjd

			imred = (imraw - medbias)/flatred
			imred = imred - np.median(imred[bgymin:bgymax, bgxmin:bgxmax])
			if not os.path.exists(path+"reduced/"):
				os.makedirs(path+'reduced/')
			im.header.append(card =('BJD_TDB', bjd[f], 'Exposure midpoint'))
			hdu = fits.PrimaryHDU(imred, header=im.header)
			hdu.writeto(path+"reduced/"+filenames[f].split('.')[0]+"_reduced.fits", output_verify='silentfix', overwrite=True)
# 			if siteloc == 'm':
# 				times[f] = (time.Time(im.header['DATE-OBS']+"T"+im.header['TIME-OBS'], format='isot', scale='utc', location=siteloc).jd - firstjd)*86400
# 				imred = (((imraw - medbias) - (meddark - medbias))/((medflat - medbias)/np.average(medflat-medbias)))[0]
#
		print("*** CENTROIDING OBJECTS, DETERMINING OPTIMAL APERTURES, AND PERFORMING PHOTOMETRY ***")
		tarx, tary, oap, oamp = optimizephot(target, imred)
		target = (int(tarx), int(tary))
		targ_flux[f] = phot(imred, oap, oap+10., tarx, tary)

		print("*** CENTROIDING COMPS, DETERMINING OPTIMAL APERTURES, AND PERFORMING PHOTOMETRY ***")
		for c in range(len(comps)):
			compx, compy, cap, camp = optimizephot(comps[c], imred)
			comps[c] = (int(compx), int(compy))
			comp_flux[c][f] = phot(imred, cap, cap+10., compx, compy)

	comp_flux = np.mean(comp_flux, axis=0)

	print("*** DIVIDING BY COMPOSITE COMPARISON AND DETRENDING WITH 2ND ORDER POLY ***")
	final = sigma_clip((targ_flux/comp_flux*np.median(comp_flux)/np.median(targ_flux)-1), sigma=3)
	cfit = np.polyfit(times, final, deg=2)
	plt.plot(times, final, '-o')
	plt.plot(times, np.polyval(cfit, times), '-o')
	plt.show()
	final = final - np.polyval(cfit, times)
	print()

	print("*** GENERATING LIGHT CURVES ***")
	os.chdir(path)
	date = im.header['DATE-OBS']
	objname = im.header['OBJECT']
	lc = open(objname+'_'+date+'.lc1', 'w')
	for n in range(len(final)):
		lc.write(str(times[n])+'\t'+str(final[n])+'\n')
	rc = open(objname+'_'+date+'.rc', 'w')
	for n in range(len(targ_flux)):
		rc.write(str(times[n])+'\t'+str(targ_flux[n])+'\n')
	bc = open(objname+'_'+date+'.bc1', 'w')
	for n in range(len(targ_flux)):
		bc.write(str(bjd[n])+'\t'+str(final[n])+'\n')

	print("*** FINISHED ***")
	print("REDUCED LC (seconds): .lc1; REDUCED LC (BJD): .bc1; RAW LC: .rc")

	plt.figure()
	plt.title(objname)
	plt.plot(times, comp_flux, label="comp")
	plt.plot(times, targ_flux, label="targ")
	plt.xlabel("Time")
	plt.ylabel("Counts")
	plt.legend()
	plt.show()

	plt.figure()
	plt.title(objname)
	plt.ylabel("Flux (%)")
	plt.xlabel("Time (BJD_TDB)")
	plt.plot(bjd, final*100., '-o')
	plt.show()

main()
