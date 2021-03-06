# -*- coding: utf-8 -*-
"""
    Created on Thu Jan 12 13:27:05 2017

    @author: zvander
    """

#############################################################
##
##  The purpose of this script is to add important
##  info into the FITS headers of SINGLE-FILTER data.
##  If you are dealing with multi-filter data, please use
##  mf_impar instead.
##
##  This script MUST be run through the command line in the
##  folder where your images are contained.  It will prompt
##  you along the way and ask whether or not you want to
##  add/edit various header keywords.
##
##  To run the entirety of this script, you will need a
##  list file, just like the ones IRAF uses, containing
##  the names of every FITS file to be operated on.  You
##  will also need a timestamps.csv file to add in the UT
##  start-of-exposure dates and times.  The timestamps file
##  is generated by Keaton Bell's OLDMAID software once it
##  has processed the entire SPE file for a given run.
##
##  Script created by Zach Vanderbosch
##  Last Update: 2017-08-01
##
#############################################################

import sys
import numpy as np
from os import getcwd
from glob import glob
from math import isnan
from astropy.io import fits
from datetime import datetime as dt
from datetime import timedelta as td
from pandas import read_csv,DataFrame

#############################################################
##
##  Progress Bar Code. I got this code from Stack Overflow,
##  "Python to print(out status bar and percentage"
##
#############################################################

## Provide the interation counter (count=int)
## and the action being performed (action=string)
def progress_bar(count,total,action):
    sys.stdout.write('\r')
    sys.stdout.write(action)
    sys.stdout.write("[%-20s] %d%%" % ('='*(count*20/total), count*100/total))
    sys.stdout.flush()
    return


#########################################################
##
##  Load in the file names which need to be parsed
##
#########################################################

path = getcwd() + '/'  # Get the current working directory

## Ask the user to supply the name of a file
## containing a list of all images to be worked on
print('')
file_list = 'ilist'
try:
    filenames  = np.loadtxt(path + file_list, dtype=str)
    print(filenames)
except IOError:
    try:
        print('Could not find a working list using default name "%s"' %file_list)
        file_list = input("Please enter the working list name to search for: ")
        filenames  = np.loadtxt(path + file_list, dtype=str)
    except IOError:
        print('Could not find a working list with name "%s"' %file_list)
        print('\nProgram TERMINATED\n')
        sys.exit(1)

## Save the original file names for comparison later
og_filenames = filenames
num_files    = len(filenames)


###############################################################
##
##  First, let's change the exposure time value in
##  the actual headers of the FITS files (EXPTIME)
##  and add in the filter information with a new
##  keyword, FILTER.
##
###############################################################

## Inform the user what values will be changed and ask whether to proceed
print('')
print('#####################################################')
print('##          This section will add and edit         ##')
print('##         the following FITS header values:       ##')
print('## ----------------------------------------------- ##')
print('## EXPTIME  - The image exposure time              ##')
print('## FILTER   - The filter in use                    ##')
print('## OBJECT   - The object name e.g. SDSSJ1529+2928  ##')
print('## INSTRUME - The instrument used. e.g. ProEM      ##')
print('## OBSERVER - The initials of the observer         ##')
print('#####################################################')
print('')
check_edit_headers = input('Would you like to proceed? (Y/[N]): ')
print('')

if (check_edit_headers == 'Y') or (check_edit_headers == 'y'):

    ## Guess the exposure time from the first frame.
    ## EXPTIME is the only header value that pre-exists
    ## from the Lightfield export-to-FITS process.
    hdu_temp  = fits.open(path + filenames[0])
    texp_read = float(hdu_temp[0].header['EXPTIME'])
    ## Check whether header exposure time is already in milliseconds or not
    ## 500ms is used as a limiting case since the shortest exposures
    ## are only ever 995ms.
    if texp_read > 500.0:
        texp_guess = round(texp_read/1000.0)
    elif texp_read < 500.0:
        texp_guess = texp_read
    hdu_temp.close()

    ## Define a function which runs the user through a
    ## prompting routine in order to change/keep a certain
    ## header keyword value.  An optional parameter is
    ## provided in case a 'best guess' at the header value
    ## is possible, such as for EXPTIME.
    def get_header_val(header_name,pass_value=None):

        ## Open the first FITS header
        hdu_temp = fits.open(path + filenames[0])

        ## Try reading the value of a header keyowrd
        try:
            ## If a pass_value was defined, use it to define header_value
            ## Else, try reading the header value from the FITS file
            if pass_value != None:
                header_value = pass_value
            else:
                ## This line throws a KeyError if the header_name
                ## keyword does not exist in the FITS header
                header_value = hdu_temp[0].header[header_name]

            ## If a current FILTER value exists, print(it and
            ## ask the user if they want to change/keep it.
            print('%s = %s' %(header_name,header_value))
            change_value = input('Change %s value? (Y/[N]): ' %header_name)

            if (change_value == 'Y') or (change_value == 'y'):
                new_value    = input('Please provide new value for %s: ' %header_name)
                header_value = new_value
                print('')
            else:
                print('%s value was not changed.' %header_name)
                print('')

        ## KeyError will be thrown if you try to read a header
        ## value that does not exist.  In this case, simply ask
        ## for a value that will be added into the header later.
        except KeyError:
            header_value = input('Please provide a value for %s: ' %header_name)

        ## Close the hdu
        hdu_temp.close()

        return header_value


    ## Use get_header_val to get values for each
    ## header keyword that needs to be added/edited
    texp        = float(get_header_val('EXPTIME', pass_value=texp_guess))
    filt_name   = get_header_val('FILTER')
    object_name = get_header_val('OBJECT')
    instr_name  = get_header_val('INSTRUME')
    observ_name = get_header_val('OBSERVER')
    print('')

    ## Let the user know what will be changed before proceeding
    print('The following values will be added to the FITS headers:')
    print('EXPTIME   = %3.1f' %texp)
    print('FILTER    = %s' %filt_name)
    print('OBJECT    = %s' %object_name)
    print('INSTRUME  = %s' %instr_name)
    print('OBSERVER  = %s' %observ_name)
    continue_edit_headers = input('Continue? (Y/[N]): ')
    print('')

    ## Defining a function to open, edit, and save a
    ## new FITS file containing the new header info
    def edit_FITS(fname, texp0, filt):
        hdu    = fits.open(fname)                    # Open FITS file
        prihdr = hdu[0].header                       # Store header info
        prihdr.set('EXPTIME' ,texp0)
        prihdr.set('FILTER'  ,filt_name  ,comment='Filter Type',before='LONGSTRN')
        prihdr.set('OBJECT'  ,object_name,comment='Object Name',before='LONGSTRN')
        prihdr.set('INSTRUME',instr_name ,comment='Instrument Name',before='LONGSTRN')
        prihdr.set('OBSERVER',observ_name,comment='Observer(s) Initials',before='LONGSTRN')
        hdu2   = fits.PrimaryHDU(hdu[0].data,prihdr) # Create new HDU
        hdu2.writeto(fname, overwrite=True)          # Write new FITS file
        hdu.close()
        return

    if (continue_edit_headers == 'Y') or (continue_edit_headers == 'y'):

        ## Loop through all FITS files in the working list
        ## and change their header values
        for i in range(num_files):

            ## print(Progress Bar
            count1  = i+1
            action1 = 'Editing header values..................'
            # progress_bar(count1, num_files, action1)

            ## Now to actually open, edit, and save the FITS files
            edit_FITS(path + filenames[i], texp, filt_name)

        print('')
        print('')
        print('FITS header values were successfully edited.')

    else:
        print('FITS headers were not changed.')

else:
    print('FITS headers were not changed.')


################################################################
##
##  Now, let's perform the task which Keatons "mcdoheader2.py"
##  performs.  That is, we need to put the absolute timestamps
##  of start-exposure into the FITS header under the keywords of
##  DATE-OBS and TIME-OBS.  A lot of the code here was taken
##  directly from Keaton's script with some slight modifications.
##
################################################################

## First load the timestamps CSV data file.
csv_name   = glob('*_timestamps.csv')
path_csv   = path + csv_name[0]
time_data  = read_csv(path_csv)

## Inform the user what values will be changed and ask whether to proceed
print('')
print('#####################################################')
print('##  This section will add and edit the following   ##')
print('##  FITS header values using a timestamps file:    ##')
print('## ----------------------------------------------- ##')
print('## DATE-OBS - UT Date at start of exposure         ##')
print('## TIME-OBS - UT Time at start of exposure         ##')
print('#####################################################')
print('')
print('Timestamps file:  %s' %csv_name[0])
print('')
check_add_times = input('Would you like to proceed? (Y/[N]): ')
print('')

if (check_add_times == 'Y') or (check_add_times == 'y'):

    ## Defining a function which gets the exposure
    ## time from the FITS header
    def get_exptime(path_to_fits):
        hdu       = fits.open(path_to_fits)
        exptime   = float(hdu[0].header['EXPTIME'])
        hdu.close()
        return exptime

    ## Defining a function to add timestamps to FITS files
    def addtimestamp(fitsname,timestamp):
        #fitsname is a string
        #timestamp is a datetime object
        hdu   = fits.open(fitsname.strip())       # Open the FITS file
        hdr   = hdu[0].header                     # Store header info
        data  = hdu[0].data                       # Extract data from HDU
        hdr.set('DATE-OBS',str(timestamp.date())) # Add DATE-OBS to header
        hdr.set('TIME-OBS',str(timestamp.time())) # Add TIME-OBS to header
        hdu2  = fits.PrimaryHDU(data,hdr)         # Create a new HDU
        hdu2.writeto(fitsname, overwrite=True, output_verify='ignore')
        hdu.close()
        return

    ## First, load the exposure times from the FITS
    ## frames and save them into a list.
    exp_times = []
    for i in range(num_files):
        ## print(progress bar
        count2  = i+1
        action2 = 'Loading Exp. Times from FITS headers...'
        # progress_bar(count2, num_files, action2)

        texp1 = get_exptime(path + filenames[i])
        exp_times.append(texp1)


    ## Perform a check to make sure the exposure times
    ## are in units of seconds, not milliseconds.  In
    ## practice, no exposure time has ever been as long
    ## as 500 seconds, but almost all exposures are longer
    ## than 500 milliseconds, so I'm using 500 as the
    ## comparison value.
    exptime_zero = exp_times[0]
    print('')
    print('First-Frame Exposure time: {} seconds.'.format(exptime_zero))
    if exptime_zero > 500:
        print('WARNING: Your exposure time value is VERY LARGE!')
        print('Make sure you have corrected the header exposure')
        print('times and converted them from milliseconds to seconds')

    ## Convert the timestamp.csv file loaded in the
    ## previous section from a Pandas data frame
    ## object to a simple Numpy matrix array.
    raw_times = DataFrame.as_matrix(time_data)

    ## Create empty lists to store data
    index   = []  # Frame tracking # for the images
    tstart  = []  # Time stamp for start of exposure
    tend    = []  # Time stamp for end of exposure
    dtstart = []  # t-delta b/t starts of current & previous frames
    dtend   = []  # t-delta b/t ends   of current & previous frames

    for line in raw_times:

        ## First split each line up into individually named values
        tindex, ttstart, ttend, tdtstart, tdtend = line
        ## Append the frame tracking # to the "index" list
        index.append(int(tindex))

        ## Next append the starting time stamp to "tstart".
        ## IF/ELSE statements check whether or not
        ## micro-seconds were included in the CSV.  It's
        ## typical for microseconds to be included unless
        ## the metadata was not saved properly.
        if len(ttstart) != 26: # If no microseconds
            tstart.append(dt.strptime(ttstart,'%Y-%m-%d %H:%M:%S'))
        else:                  # If there are micro-seconds
            tstart.append(dt.strptime(ttstart,'%Y-%m-%d %H:%M:%S.%f'))

        ## Next append the ending time stamp to "tend"
        if len(ttend) != 26:   # If no microseconds
            tend.append(dt.strptime(ttend,'%Y-%m-%d %H:%M:%S'))
        else:                  # If there are microseconds
            tend.append(dt.strptime(ttend,'%Y-%m-%d %H:%M:%S.%f'))

        ## Lastly, append the start-to-start and end-to-end
        ## delta-t values between adjacent exposures.
        if isnan(tdtstart) == False: # If not the first frame
            dtstart.append(float(tdtstart)/1e9)
            dtend.append(float(tdtend)/1e9)
        else:                            # If the first frame
            dtstart.append(exptime_zero)
            dtend.append(exptime_zero)


    ## All times are defined relative to first frame
    tzero = tstart[0]
    ms    = tzero.microsecond # Number of microseconds in time stamp

    ## Any reason to worry that GPS triggering was not used?
    ## This IF statement checks whether the first time stamp
    ## came more than 0.05 seconds before or after an
    ## integer second.
    if (np.abs(ms - 1e6) > 5e4) & (np.abs(ms) > 5e4):
        print("WARNING: First exposure > 0.05 seconds away from integer second.")
        print("Check that you were using GPS triggers.")

    ## Round tzero to the nearest second
    ms = tzero.microsecond
    if ms > 5e5: #round up
        tzero += td(microseconds = 1e6 - ms)
    else:        #round down
        tzero += td(microseconds = -1 * ms)

    ## Create a list to hold the final timestamp values
    ## which will then be added into the FITS headers
    times = [tzero]
    #Add first timestamp to first fits file
    addtimestamp(path + filenames[0],times[0])

    #Determine accurate timestamps and place in fits headers.
    for i in range(1,num_files):

        ## print(progress bar
        count3  = i+1
        action3 = 'Adding timestamps to FITS headers......'
        # progress_bar(count3, num_files, action3)

        ## Get the exposure time for the current frame
        exptime = exp_times[i]

        ## Check that the expected time has elapsed.
        ## Unfortunately, there's an indexing issue here.
        ## The exp-time for one frame has to be compared
        ## to the "dtstart" of the next frame.  This works
        ## fine until the last frame when there is no
        ## longer a dtstart to compare to.  In that case.\,
        ## the "elif" statement below, I just compare to
        ## the "dtend" of the same frame and hope for the best.
        if   (i < num_files-1  and round(exptime) == round(dtstart[i+1])):
            times.append(times[i-1] + td(seconds = round(dtstart[i])))
        elif (i == num_files-1 and round(exptime) == round(dtend[i])):
            times.append(times[i-1] + td(seconds = round(dtstart[i])))
        else:
            print('')
            print('')
            print("WARNING: timestamp anomaly on frame {}".format(index[i]))
            ## Sometimes a bad timestamp comes down the line and
            ## is corrected on the next exposure. Check to see if
            ## things get back on track and the exposure time was
            ## really the expected duration.
            ## WARNING! The checks below may or may not work for
            ## multi-filter data, yet to be confirmed.
            if i < len(index)-3:

                dt_check1  = dtstart[i]+dtstart[i+1]+dtstart[i+2]
                dt_check2  = dtstart[i-1]+dtstart[i]+dtstart[i+1]
                exp_check1 = exp_times[i]+exp_times[i+1]+exp_times[i+2]
                exp_check2 = exp_times[i-1]+exp_times[i]+exp_times[i+1]

                if round(dt_check1) == round(exp_check1):
                    print("It appears to get back on track.")
                    times.append(times[i-1] + td(seconds = round(exptime)))
                elif round(dt_check2) == round(exp_check2):
                    print("Making up for the last frame.")
                    times.append(times[i-1] + td(seconds = round(exptime)))
                else:
                    print("Looks like triggers were missed.")
                    times.append(times[i-1] + td(seconds = round(dtstart[i])))
            else:
                print("Last couple of timestamps from this run are suspect.")
                times.append(times[i-1] + td(seconds = round(exptime)))

        ## Add timestamp to fits file:
        addtimestamp(path + filenames[i],times[-1])

    print('')
    print('')
    print('Successfully added UT timestamps to FITS headers.')
    print('')

else:
    print('Timestamps were not added to the FITS headers.')
    print('')
