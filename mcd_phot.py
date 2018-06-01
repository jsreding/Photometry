import numpy as np
import sys
import pyfits
import glob
import os
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.stats import sigma_clip
import csv
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
os.chdir("../../bias")
biases = glob.glob("bias*.fits")
medbias = medcomb(biases)
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
exptime = int(pyfits.open(filenames[0])[0].header['EXPTIME'])
os.chdir("../../dark")
darks = glob.glob("dark_"+str(exptime)+"*.fits")
meddark = medcomb(darks)
os.chdir(path)

#Find appropriate flats, median combine
flatfilt = pyfits.open(filenames[0])[0].header['FILTER']
flatfilt = flatfilt.replace(" ", "")
if flatfilt == "BG40":
	flatfilt = flatfilt.replace("BG", "bg")
os.chdir("../../dome_flat")
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

targ_flux = np.zeros(num_files)
comp_flux = np.zeros((len(comps), num_files))
time = np.zeros(num_files)
for f in range(len(filenames)):
	im = pyfits.open(filenames[f])[0]
	imraw = im.data
	time[f] = Time(im.header['DATE-OBS']+"T"+im.header['TIME-OBS'], format='isot', scale='utc').jd
	imred = (((imraw - medbias) - (meddark - medbias))/((medflat - medbias)/np.average(medflat-medbias)))[0]

	if not os.path.exists(path+"reduced/"):
		os.makedirs(path+'reduced/')
	hdu = pyfits.PrimaryHDU(imred)
	hdu.writeto(path+"reduced/"+filenames[f]+"_reduced.fits", clobber=True)

	tarx, tary = np.where(imred == imred[target[1]-5:target[1]+5, target[0]-5:target[0]+5].max())[1][0], np.where(imred == imred[target[1]-5:target[1]+5, target[0]-5:target[0]+5].max())[0][0]
	targ_flux[f] = float(phot(5, 10, tarx, tary))

	for c in range(len(comps)):
		compx, compy = (np.where(imred == imred[comps[c][1]-5:comps[c][1]+5, comps[c][0]-5:comps[c][0]+5].max())[1][0], np.where(imred == imred[comps[c][1]-5:comps[c][1]+5, comps[c][0]-5:comps[c][0]+5].max())[0][0])
		comp_flux[c][f] = float(phot(5, 8, compx, compy))

comp_flux = np.mean(comp_flux, axis=0)
final = sigma_clip(np.array(targ_flux)/np.array(comp_flux)*np.median(np.array(comp_flux)), sigma=4)

date = pyfits.open(filenames[0])[0].header['DATE-OBS']
# os.remove(path.split('/')[-2]+'_'+date+'.csv')
lc = open(path.split('/')[-2]+'_'+date+'.lc1', 'w')
for n in range(len(final)):
	lc.write(str(final[n])+'\t'+str(time[n])+'\n')

plt.figure()
plt.title(path.split('/')[-2])
plt.ylabel("Flux (raw counts)")
plt.xlabel("Time (JD)")
plt.scatter(time, final)
plt.plot(time, final)
plt.show()
