import numpy as np
import sys
import astropy.io.fits as pyfits
import glob
import os
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.stats import sigma_clip
from photutils import CircularAperture, CircularAnnulus, aperture_photometry, centroid_com
import csv
from ft import fft

def medcomb(files):
	arr = []
	for f in files:
		arr.append(pyfits.open(f)[0].data)
	arr = np.median(arr, axis=0)
	return arr

def phot(R1, R2, objx, objy):
	ap = CircularAperture((objx, objy), R1)
	print(ap)
	ann = CircularAnnulus((objx, objy), R1, R2)
	apers = [ap, ann]
	phot = aperture_photometry(imred, apers)
	print(phot)
	bg = float(phot['aperture_sum_1'])/ann.area()*ap.area()
	print(bg)
	sub = float(phot['aperture_sum_0']) - bg
	print(sub)
	return sub

path = os.getcwd() + '/'

#read in biases, median combine
os.chdir("../bias")
if "mcdonald" in path:
	biases = glob.glob("bias*.fits")
elif "goodman" in path:
	biases = glob.glob("*zero*.fits")
medbias = medcomb(biases)
hdu = pyfits.PrimaryHDU(medbias)
hdu.writeto("master_bias.fits", overwrite=True)

os.chdir(path)

#Get list of filenames
l = input("Image list: ")
try:
	filenames  = np.loadtxt(path + l, dtype=str)
except IOError:
	print('Could not find a working list with name "%s"' %file_list)
	print('\nProgram TERMINATED\n')
	sys.exit(1)
num_files = len(filenames)

#Find appropriate darks, median combine
if "mcdonald" in path:
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
if "mcdonald" in path:
	flats = glob.glob("dome_flat_"+flatfilt+"*.fits")
elif "goodman" in path:
	flats = glob.glob("*DomeFlat*")
medflat = medcomb(flats)
flatred = (medflat - medbias)/np.mean(medflat-medbias)
hdu = pyfits.PrimaryHDU(flatred)
hdu.writeto("master_flat.fits", overwrite=True)

os.chdir(path)

#Find target and comparison centers
sources = np.loadtxt('phot_coords.orig', dtype=float)
target = (int(sources[0][0]), int(sources[0][1]))
comps = (int(sources[1][0]), int(sources[1][1]))
# for s in range(1, len(sources)):
	# comp = sources[s]
	# comps.append((int(comp[0]), int(comp[1])))

targ_flux = np.zeros(num_files)
comp_flux = np.zeros(num_files)
time = np.zeros(num_files)
im0 = pyfits.open(filenames[0])[0]
if "mcdonald" in path:
	firstjd = Time(im0.header['DATE-OBS']+"T"+im0.header['TIME-OBS'], format='isot', scale='utc').jd
elif "goodman" in path:
	firstjd = Time(im0.header['DATE-OBS'], format='isot', scale='utc').jd
for f in range(len(filenames)):
	im = pyfits.open(filenames[f])[0]
	imraw = im.data
	if "mcdonald" in path:
		time[f] = (Time(im.header['DATE-OBS']+"T"+im.header['TIME-OBS'], format='isot', scale='utc').jd - firstjd)*86400
		imred = (((imraw - medbias) - (meddark - medbias))/((medflat - medbias)/np.average(medflat-medbias)))[0]
	elif "goodman" in path:
		time[f] = (Time(im.header['DATE-OBS'], format='isot', scale='utc').jd - firstjd)*86400
		imred = (imraw - medbias)/flatred

	if not os.path.exists(path+"reduced/"):
		os.makedirs(path+'reduced/')
	hdu = pyfits.PrimaryHDU(imred)
	hdu.writeto(path+"reduced/"+filenames[f].split('.')[0]+"_reduced.fits", overwrite=True)

	print(target)

	tarx, tary = centroid_com(imred[target[1]-5:target[1]+5, target[0]-5:target[0]+5])
	tarx = tarx + target[0]-5
	tary = tary + target[1]-5
	print("targ location:", tarx, tary)
	targ_flux[f] = float(phot(5, 10, tarx, tary))

	# for c in range(len(comps)):
	compx, compy = centroid_com(imred[comps[1]-5:comps[1]+5, comps[0]-5:comps[0]+5])
	compx = compx + comps[0]-5
	compy = compy + comps[1]-5
	print("comp location:", compx, compy)
	comp_flux[f] = float(phot(5, 10, compx, compy))

# comp_flux = np.mean(comp_flux, axis=0)
final = sigma_clip(targ_flux/comp_flux*np.median(comp_flux)/np.median(targ_flux)-1, sigma=3)

date = pyfits.open(filenames[0])[0].header['DATE-OBS']
# os.remove(path.split('/')[-2]+'_'+date+'.csv')
lc = open(path.split('/')[-2]+'_'+date+'.lc1', 'w')
for n in range(len(final)):
	lc.write(str(time[n])+'\t'+str(final[n])+'\n')
rc = open(path.split('/')[-2]+'_'+date+'.rc', 'w')
for n in range(len(targ_flux)):
	rc.write(str(time[n])+'\t'+str(targ_flux[n])+'\n')

plt.figure()
plt.title(path.split('/')[-2])
plt.ylabel("Flux (%)")
plt.xlabel("Time (JD)")
plt.scatter(time, final*100.)
plt.plot(time, final*100)
plt.show()
