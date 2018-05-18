import numpy as np
import sys
import pyfits
import glob

#read in bias
bs = glob.glob()

#read in dark and reduce
dk = pyfits.open(sys.argv[2])
if ('BIASCORR' in dk[0].header) == False:
	dk_bs = (dk[0].data - bs[0].data)
else:
	dk_bs = dk[0].data

#create dictionaries for stacked image alignment
maxdata_med = {}
maxdata_avg = {}

#Start loop to read filters
imagenames = glob.glob('*_*_???.fits')
flatnames = glob.glob('*_mflat_*.fits')
filts = set([x.split('_')[2] for x in imagenames])
for flt in filts:
	imnames = glob.glob('ngc*' + flt + '*.fits')
	ftnames = glob.glob('*mflat*' + flt + '*.fits')
	#read in one image for exposure time
	lt = pyfits.open(imnames[0])

	#calibrate dark
	dk_red = dk_bs * (lt[0].header['EXPOSURE']/dk[0].header['EXPOSURE'])

	#read in flat and reduce
	for flat in ftnames:
		ft = pyfits.open(flat)
		if ('BIASCORR' in ft[0].header) == False:
			ft_bs = ft[0].data - bs[0].data
		else:
			ft_bs = ft[0].data
		if ('DARKCORR' in ft[0].header) == False:
			ft_red = (ft_bs - dk_red)/np.average((ft_bs - dk_red))
		else:
			ft_red = ft_bs/np.average(ft_bs)

	#reduce test image, find location of max
	if ('BIASCORR' in lt[0].header) == False:
			lt_bs = lt[0].data - bs[0].data
	else:
		lt_bs = lt[0].data
	if ('DARKCORR' in lt[0].header) == False:
		lt_red = lt_bs - dk_red
	else:
		lt_red = lt_bs
	lt_data = (lt_red - dk_red) / ft_red
	#choosing near image center for brightest point determination
	x_center = len(lt_data)/2
	y_center = len(lt_data)/2
	ltmax = np.where(lt_data==np.max(lt_data[x_center-100:x_center+100, y_center-100:y_center+100]))

	#read in images and reduce
	images = []
	for img in imnames:
		i = pyfits.open(img)
		if ('BIASCORR' in i[0].header) == False:
			i_bs = i[0].data - bs[0].data
		else:
			i_bs = i[0].data
		if ('DARKCORR' in i[0].header) == False:
			i_red = i_bs - dk_red
		else:
			i_red = i_bs
		img_data = (i_red - dk_red) / ft_red
		imax = np.where(img_data==np.max(img_data[x_center-100:x_center+100, y_center-100:y_center+100]))
		xshift = ltmax[1] - imax[1]
		yshift = ltmax[0] - imax[0]
		img_data = np.roll(img_data, xshift, 1)
		img_data = np.roll(img_data, yshift, 0)
		images.append(img_data)
	image_array = np.array(images)

	#Average images, align to blue
	img_avg = np.average(image_array, axis = 0)
	maxdata_avg[flt] = np.where(img_avg==np.max(img_avg[x_center-100:x_center+100, y_center-100:y_center+100]))
	avg_xshift = maxdata_avg['Blue'][1] - maxdata_avg[flt][1]
	avg_yshift = maxdata_avg['Blue'][0] - maxdata_avg[flt][0]
	img_avg = np.roll(img_avg, avg_xshift, 1)
	img_avg = np.roll(img_avg, avg_yshift, 0)
	hdu_avg = pyfits.PrimaryHDU(img_avg)
	hdu_avg.writeto(flt+'_avg.fits', clobber=True)

	#Median images, align to blue
	img_med = np.median(image_array, axis = 0)
	maxdata_med[flt] = np.where(img_med==np.max(img_med[x_center-100:x_center+100, y_center-100:y_center+100]))
	med_xshift = maxdata_med['Blue'][1] - maxdata_med[flt][1]
	med_yshift = maxdata_med['Blue'][0] - maxdata_med[flt][0]
	img_med = np.roll(img_med, med_xshift, 1)
	img_med = np.roll(img_med, med_yshift, 0)
	hdu_med = pyfits.PrimaryHDU(img_med)
	hdu_med.writeto(flt+'_med.fits', clobber=True)
