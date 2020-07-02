#define a few things that are common to all three populations in the BMC calcuation.
#This includes the simpsons rule function, and likelihood functions.
import sys
import os
import warnings
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
def _J_filter(filename,testingmode=0):
    d = csv_load(filename)
    if testingmode:
        return "J"
    #check for Euclid first as the integtation range can be bigger
    elif len([c for c in d.columns if c.startswith('J_EUC')])>0:
        return 'EUC'
    elif len([c for c in d.columns if c.startswith('J_VIK')])>0:
        return 'MKO_VIK'
    elif len([c for c in d.columns if c.startswith('J_LAS')])>0:
        return 'MKO_LAS'
    elif len([c for c in d.columns if c.startswith('J_UV')])>0:
        return 'MKO_UV'
    #    print("MKO J band found. Is this data from VIKING? (This affects the ETG prior)")
    #    viking = input("y/n:")
    #    if viking == "y":
    #        return 'MKO_VIK'
    #    elif viking == "n":
    #        return "MKO"
    #    else:
    #        print("Choice not understood. Try again.")
    #        return _J_filter(filename)
    else:
        print("Error: Euclid or MKO (either VIKING, UKIDSS LAS or COSMOS ULtraVISTA) J band need to be present and correctly labelled")
        raise SystemExit

#initialise values for luptitudes
def _m0_values_setup():
    m0 = { 'i_SDSS': 24.36181874,'z_SDSS': 22.8269207,\
    'VIK': {'Z_MKO': 26.104,'Y_MKO': 25.225,'J_MKO': 24.411,'H_MKO': 23.660,'KS_MKO':23.040}}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fileband, filem0 = np.genfromtxt(startpath+'/options/asinh_m0.values',usecols=(0,1),dtype=str,unpack=True)
        for i,f in enumerate(fileband):
            if f.endswith("SDSS"):
                print(f"Asinh mags warning: new m0 value specified for {f}. Overriding default SDSS m0 value.") 
                m0[f] = float(filem0[i])
            elif f.endswith("VIK"):
                us = f.find("_")
                print(f"Asinh mags warning: new m0 value specified for {f}. Overriding default VIKING m0 value.")
                m0['VIK'][f[:us]+"_MKO"] = float(filem0[i])
            elif f.endswith("LAS"):
                us = f.find("_")
                if 'LAS' not in m0:
                    m0['LAS'] = {f[:us]+"_MKO":float(filem0[i])}
                else:
                    m0['LAS'][f[:us]+"_MKO"] = float(filem0[i])
            elif f.endswith("UV"):
                us = f.find("_")
                if 'UV' not in m0:
                    m0['UV'] = {f[:us]+"_MKO":float(filem0[i])}
                else:
                    m0['UV'][f[:us]+"_MKO"] = float(filem0[i])
            else:
                m0[f] = float(filem0[i])
    except ValueError:
        pass
    return m0

########################################################################################################################################

#photometric conversions
        
########################################################################################################################################

def _flux2mag(flx,f0):
    return -2.5*np.log10(flx/f0)
        
def _mag2flux(mag,f0):
    return f0 * (10.**(-0.4*mag))

def _magerr2fluxerr(mag,magerr,f0):
    flux = _mag2flux(mag,f0)
    return flux * magerr * 2.3 / 2.5

#
def _luptitude_m0(band,mko):
    if mko:
        us = band.find("_")
        try:
            return _m0_values[mko][band[:us]+"_MKO"]
        except KeyError:
            print("Error: only SDSS and VIKING m0 values are available by default for asinh magnitudes")
            print(f"m0 value for {band[:us]+mko} needs to be specified in /options/asinh_m0.values")
            print("Exiting...")
            raise SystemExit
    else:
        try:
            return _m0_values[band]
        except KeyError:
            print("Error: only SDSS and VIKING m0 values are available by default for asinh magnitudes")
            print(f"m0 value for {band} needs to be specified in /options/asinh_m0.values")
            print("Exiting...")
            raise SystemExit
            
def _lup2flux(m,m0,f0):
    a = 2.*f0*(10.**(-0.4*m0))
    b = 0.4*np.log(10)*(m0-m)
    return a*np.sinh(b)

def _luperr2fluxerr(m,sigm,m0,f0):
    #print(m,sigm,m0,f0)
    a = 0.8 * np.log(10) * f0 * (10.**(-0.4*m0)) #* sigm 
    b = -0.4 * np.log(10) * (m0-m)
    #print(_lup2flux(m,m0,f0))
    #print(a*np.cosh(b)*sigm)
    return a*np.cosh(b)*sigm

def _luperr2fluxerr2(m,sigm,m0,f0):
    #print(m,sigm,m0,f0)
    f = _lup2flux(m,m0,f0); print(f)
    a = sigm*np.log(10)/2.5
    b = 2*f0*10.**(-0.4*m0)
    c = np.sqrt(f**2. + b**2.)
    #print(a*c)
    return a*c

########################################################################################################################################

#sinb calculation - could this be moved to mlt.py?

########################################################################################################################################



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

########################################################################################################################################

#process the file

########################################################################################################################################      
        
#_sinb above checks the first 2 columns are indeed position
#this function should check the photometry columns are consistent
def _check_photometry(d):
    fcols = [col for col in d.index if "_flux" in col or "_ab" in col or "_vega" in col]
    ecols = [col for col in d.index if "_err" in col]
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
        label = colpairs[p][0]
        us2 = label.find("_",3)
        band = label[:us2]
        mko = 0
        #deal with the different MKO options
        if band.endswith("VIK") or band.endswith("LAS") or band.endswith("UV"):
            us1 = band.find("_")
            mko = band[us1+1:]
            band = band[:us1] + "_MKO" 
            
            
        if band not in _vega_zp: #check we have colours etc for this band
            print(f"{band} not supported. Calculation will continue without this band.")
            continue
        #turn each pair into flux and error
        #check if the entry is a flux, ab or vega mag or luptitude
        #we also check for nans and strings (=nsig) before trying conversions
        if "_flux" in label:
            f = d[colpairs[p][0]]
            e = d[colpairs[p][1]]
        elif "mag" in label:
            if "_abmag" in label:
                ZP = 3631.
            elif "_vegamag" in label:
                ZP = _vega_zp[band]
            else:
                print(f"Unable to interpret {band} magnitude type. Calculation will continue without this band.")
                continue
            #f & e columns
            if pd.isnull(d[label]):
                f = d[label]
            else:
                f = _mag2flux(d[label],ZP)
            #e
            if pd.isnull(d[colpairs[p][1]]) or (type(d[colpairs[p][1]]) == str and d[colpairs[p][1]].endswith('sig')):
                e = d[colpairs[p][1]]
                # type(d[colpairs[p][1]]) == str and not d[colpairs[p][1]].endswith('sig'):
            else:
                #if there are 'nsig' entries elsewhere in the column, any full uncertainty measurments will be strings
                #simplest to apply the float conversion to e in all cases
                e =  _magerr2fluxerr(d[colpairs[p][0]],float(d[colpairs[p][1]]),ZP)
            #else:
            #    e = _magerr2fluxerr(d[colpairs[p][0]],d[colpairs[p][1]],ZP)
        elif "lup" in label:
            if "_ablup" in label:
                ZP = 3631.
                #if "z_SDSS" in label:
                #    print(d[label])
                #    try:
                #        d[label] -= 0.00
                #    except (TypeError, ValueError):
                #        pass
            elif "_vegalup" in label:
                ZP = _vega_zp[band]
            else:
                print(f"Unable to interpret {band} luptitude type. Calculation will continue without this band.")
                continue
            m0 = _luptitude_m0(band,mko)
            #f&e columns
            #f
            if pd.isnull(d[label]):
                f = d[label]
            else:
                f = _lup2flux(d[label],m0,ZP)
            #e
            if pd.isnull(d[colpairs[p][1]]) or (type(d[colpairs[p][1]]) == str and d[colpairs[p][1]].endswith('sig')):
                e = d[colpairs[p][1]]
            #elif type(d[colpairs[p][1]]) == str and not d[colpairs[p][1]].endswith('sig'):    
            else: 
                #float here for same reason as in the mag err stuff above
                e = _luperr2fluxerr(d[colpairs[p][0]],float(d[colpairs[p][1]]),m0,ZP)     
        
        #now make an entry in the dictionary for f&e - only if values exist
        #here f = fstring, not flux!
        if np.isfinite(f):
            entryname = band+f"{nsi}"
            #if this is the first time a band is encountered just put it in the dictionary
            #if entryname in fluxdict:
            while entryname in fluxdict:
                nsi += 1
                entryname = band+f"{nsi}"
            fluxdict[entryname] = {'f':f,'e':e}
    if fluxdict: #empty dictionary is bad! it evaluates to false
        #print(fluxdict)
        return fluxdict
    else:
        print("Error! Unable to find any valid bands to work with. Check photometry file.")
        print("Exiting...")
        raise SystemExit
        
        
########################################################################################################################################

#a few other definitions
        
########################################################################################################################################

if __name__ == "__main__":
    print("Value of __name__ is:", __name__)
    print("Running dataload.py module")
    startpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _vega_zp = {'i_PS': 2577.,'z_PS': 2273.,'y_PS': 2204.,\
            #note LSST & PS are very similar.
            'i_LSST':2578.,'z_LSST': 2272.,'y_LSST': 2187.,\
            'i_SDSS': 2602.,'z_SDSS': 2245.,\
            'z_DEC': 2247.,'Y_DEC': 2146.,\
            'z_COS': 2253.,'y_COS': 2186.,\
            'Z_MKO': 2247.,'Y_MKO': 2055.,'J_MKO': 1531.,'H_MKO': 1014.,'K_MKO':631.,'KS_MKO':667.,\
            'W1_WISE':310.,'W2_WISE':170.,\
            'O_EUC': 2667.,'Y_EUC': 1900.,'J_EUC': 1354.,'H_EUC': 922.}
    
    
    filename = "sdss2/data.csv"
    _m0_values = _m0_values_setup()
    d = csv_load(filename)
    
    jf = _J_filter(filename,testingmode=1)
    d.apply(lambda x: _flux_dict(x), axis=1)
    #do other test stuff...
else:
    startpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        filename = sys.argv[1]
    except IndexError: #should only occur if pqrun/plots are NOT run from command line - testing assumed.
        filename = "J1120.sed"
        #filename = "vikingquasar.flx"
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
    
       
    _m0_values = _m0_values_setup()
    
    
    #5sigma depths for qsfs which will be added in future.
    _depth_5sig_ab = {'i_PS': 23.1,'z_PS': 22.3,'y_PS': 21.3,\
            'i_LSST':25.6,'z_LSST': 24.9,'y_LSST': 23.7,\
            'i_SDSS': 22.2,'z_SDSS': 20.7,\
            'z_DEC': 23.5,'Y_DEC': 22.2,\
            'z_COS': 25.3,'y_COS': 23.8,\
            'Z_MKO': {'LAS':np.nan,'VIK':22.6},'Y_MKO': {'LAS': 20.8,'VIK':21.9},\
            'J_MKO': {'LAS': 20.4,'VIK':21.7},'H_MKO': {'LAS':20.1,'VIK':21.1},\
            'K_MKO': {'LAS': 20.,'VIK':np.nan},'KS_MKO': {'LAS':np.nan,'VIK':21.1},\
            'W1_WISE':19.3,'W2_WISE':18.9,\
            'O_EUC': 25.3,'Y_EUC': 24.,'J_EUC': 24.,'H_EUC': 24.}
    
    if sys.argv[0].endswith("plots.py") or sys.argv[0].endswith("output.py"):
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
      