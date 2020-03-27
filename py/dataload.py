#define a few things that are common to all three populations in the BMC calcuation.
#This includes the simpsons rule function, and likelihood functions.
import sys
import os
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
#import pqrun as pqr

########################################################################################################################################

#Load the data file, extract the relevant filters etc.

########################################################################################################################################

def csv_load(filename):
    d = pd.read_csv(startpath+"/data/"+filename)
    return d

def csv_save(d):
    if outname == 0:
        d.iloc[:,:].to_csv(startpath+"/data/"+filename,index=0)
        return f"Results have been appended to {startpath}/data/{filename}"
    else:
        lastslashix = outname.rfind("/")
        if lastslashix > -1:
            outfolder = startpath+"/data/"+outname[:lastslashix]
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)
        d.iloc[:,:].to_csv(startpath+"/data/"+outname,index=0)
        return f"Results have been written to {startpath}/data/{filename}"

#establish which filters are used in a given dataset
def _J_filter(filename):
    d = csv_load(filename)
    #check for Euclid first as the integtation range can be bigger
    if len([c for c in d.columns if c.startswith('J_EUC')])>0:
        return 'EUC'
    elif len([c for c in d.columns if c.startswith('J_MKO')])>0:
        print("MKO J band found. Is this data from VIKING? (This affects the ETG prior)")
        viking = input("y/n:")
        if viking == "y":
            return 'MKO_VIK'
        elif viking == "n":
            return "MKO"
        else:
            print("Choice not understood. Try again.")
            return _J_filter(filename)
    else:
        print("Error: Either Euclid or MKO J band need to be present and correctly labelled")
        raise SystemExit

def _flux2mag(flx,zp):
    return -2.5*np.log10(flx/zp)
        
def _mag2flux(mag,zp):
    return zp * (10.**(-0.4*mag))

def _magerr2fluxerr(mag,magerr,zp):
    flux = _mag2flux(mag,zp)
    return flux * magerr * 2.3 / 2.5


#return the magnitude of sinb for the MLT population
def _getsinb(r,d):
    ra = np.deg2rad(r)
    dec = np.deg2rad(d)
    return abs((np.sin(dec)*np.cos(1.093)) - (np.cos(dec)*np.sin(ra - 4.926)*np.sin(1.093)))
        
#return absolute value of sin b for a row.      
def _sinb(d):
    if 'ra_d' in d and 'dec_d' in d.columns[:2]:
        d['sinb'] = d.apply(lambda x: _getsinb(x.ra_d,x.dec_d), axis=1)
    elif 'ra_s' in d and 'dec_s' in d.columns[:2]:
        d['ra_d'] = d.apply(lambda x: SkyCoord(x.ra_s,x.dec_s, unit=(u.hourangle, u.deg)).ra.deg, axis=1)
        d['dec_d'] = d.apply(lambda x: SkyCoord(x.ra_s,x.dec_s, unit=(u.hourangle, u.deg)).dec.deg, axis=1)
        d['sinb'] = d.apply(lambda x: _getsinb(x.ra_d,x.dec_d), axis=1)
    else:
        print('Error: please specify ra and dec in the first two columns of the data file,\
              either in decimal degrees or hh:mm:ss.ss dd:mm:ss.ss format')
        raise SystemExit
    return d

        
        
#_sinb above checks the first 2 columns are indeed position
#this function should check the photometry columns are consistent
def _check_photometry(d):
    fcols = [col for col in d.index if "_f" in col or "_v" in col or "_a" in col]
    ecols = [col for col in d.index if "_e" in col]
    if len(fcols) != len(ecols):
        print('Error: number of flux columns does not match number of uncertainty columns')
        print('Please check data input file and try again')
        raise SystemExit
    elif len(fcols) == 0:
        print('Error: no valid photometry columns.')
        print('Please check data input file and try again')
        raise SystemExit
    #check the columns are paired correctly
    #an important point in each column heading is the second underscore. Preceding this is the filter,
    #after this is extra photometry information.
    #zcols = zipped pairs of column names, photometry and uncertainty 
    zcols = list(zip(fcols,ecols))
    for p in range(len(zcols)):
        #starting the search at index 3 will definitely pick up the second underscore given the filter possibilities
        us2 = zcols[p][0].find("_",3)
        #check the same band is used in each pair of f/e columns
        if zcols[p][0][:us2] != zcols[p][1][:us2]:
            print('Error: labels of flux and uncertainty columns do not correspond.')
            print(f'Please check data input file starting at column {p+2} and try again')
            raise SystemExit
    return zcols

#return a dictionary of fluxes for a row of photometry
def _flux_dict(d):
    fluxdict = {}
    #this should return matching pairs of photom and uncertainty we can iterate over
    colpairs = _check_photometry(d)
    for p in range(len(colpairs)):
        nsi = 1
        us2 = colpairs[p][0].find("_",3)
        #turn each pair into flux and error
        #check if the entry is a flux, ab or vega mag
        #we also check for nans and strings (=nsig) before trying conversions
        if colpairs[p][0][us2:us2+2] == "_f":
            f = d[colpairs[p][0]]
            e = d[colpairs[p][1]]
        elif colpairs[p][0][us2:us2+2] == "_a":
            #f
            if pd.isnull(d[colpairs[p][0]]):
                f = d[colpairs[p][0]]
            else:
                f = _mag2flux(d[colpairs[p][0]],3631.)
            #e
            if pd.isnull(d[colpairs[p][1]]) or (type(d[colpairs[p][1]]) == str and d[colpairs[p][1]].endswith('sig')):
                e = d[colpairs[p][1]]
            elif type(d[colpairs[p][1]]) == str and not d[colpairs[p][1]].endswith('sig'):
                e =  _magerr2fluxerr(d[colpairs[p][0]],float(d[colpairs[p][1]]),3631.)
            else:
                e = _magerr2fluxerr(d[colpairs[p][0]],d[colpairs[p][1]],3631.)
        elif colpairs[p][0][us2:us2+2] == "_v":
            ZP = _vega_zp[colpairs[p][0][:us2]]
            #f
            if pd.isnull(d[colpairs[p][0]]):
                f = d[colpairs[p][0]]
            else:
                f = _mag2flux(d[colpairs[p][0]],3631.)
            #e
            if pd.isnull(d[colpairs[p][1]]) or (type(d[colpairs[p][1]]) == str and d[colpairs[p][1]].endswith('sig')):
                e = d[colpairs[p][1]]
            elif type(d[colpairs[p][1]]) == str and not d[colpairs[p][1]].endswith('sig'):
                e =  _magerr2fluxerr(d[colpairs[p][0]],float(d[colpairs[p][1]]),ZP)
            else:
                e = _magerr2fluxerr(d[colpairs[p][0]],d[colpairs[p][1]],ZP)
        #now make an entry in the dictionary for f&e - only if values exist
        if np.isfinite(f):
            entryname = colpairs[p][0][:us2]+f"{nsi}"
            #if this is the first time a band is encountered just put it in the dictionary
            if entryname in fluxdict:
                while entryname in fluxdict:
                    nsi += 1
                    entryname = entryname[:-1]+f"{nsi}"
            fluxdict[entryname] = {'f':f,'e':e}
    return fluxdict

if __name__ == "__main__":
    print("Value of __name__ is:", __name__)
    print("Running dataload.py module")
    startpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = "test/z_2siglims.data"
    #do other test stuff...
else:
    startpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        filename = sys.argv[1]
    except IndexError: #should only occur if pqrun/plots are NOT run from command line - testing assumed.
        filename = "Banados_quasar.data"
    _Jfilt = _J_filter(filename)      
#vega zero points in Jy. Combination of SVO filter sevice, Hewett+ 2006 and CASU homepage.
    _vega_zp = {'i_PS': 2577.,'z_PS': 2273.,'y_PS': 2204.,\
            #note LSST & PS are very similar.
            'i_LSST':2578.,'z_LSST': 2272.,'y_LSST': 2187.,\
            'i_SDSS': 2602.,'z_SDSS': 2245.,\
            'z_DEC': 2247.,'Y_DEC': 2146.,\
            'z_COS': 2253.,'y_COS': 2186.,\
            'Z_MKO': 2247.,'Y_MKO': 2055.,'J_MKO': 1531.,'H_MKO': 1014.,'K_MKO':631.,'KS_MKO':667.,\
            'W1_WISE':310.,'W2_WISE':170.,\
            'O_EUC': 2667.,'Y_EUC': 1900.,'J_EUC': 1354.,'H_EUC': 922.}
    
    if sys.argv[0].endswith("plots.py"):
        print("Diagnostic code running. We will be producing plots for specified candidates.")
        #effective wavelength in microns, all from SVO filter service
        _eff_wavelength = {'i_PS': 0.750,'z_PS': 0.867,'y_PS': 0.961,\
            #note LSST & PS are very similar.
            'i_LSST':0.750,'z_LSST': 0.868,'y_LSST': 0.971,\
            'i_SDSS': 0.744,'z_SDSS': 0.890,\
            'z_DEC': 0.916,'Y_DEC': 0.988,\
            'z_COS': 0.909,'y_COS': 0.975,\
            'Z_MKO': 0.877,'Y_MKO': 1.020,'J_MKO': 1.252,'H_MKO': 1.645,'K_MKO':2.201,'KS_MKO':2.147,\
            'W1_WISE':3.353,'W2_WISE':4.603,\
            'O_EUC': 0.687,'Y_EUC': 1.073,'J_EUC': 1.343,'H_EUC': 1.734}
        
    elif sys.argv[0].endswith("pqrun.py"):
        print("Probability code running. Let's try to find some high-redshift quasars!")
        try:
            outname = sys.argv[2]
        except IndexError:
            outname = 0       
      