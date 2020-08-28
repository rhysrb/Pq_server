#python 3
#this script is part of the Pq server code
#it sets up the high-redshift quasar population
#the relevant cosmology functions are also defined here.
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import shelve
import pandas as pd
#custom modules
import dataload as dld
import bmc_common as bmccom

########################################################################################################################################

#Produce lookup tables for cosmology

########################################################################################################################################

def _check_cosmology():
    print("Checking cosmology options file...")
    setting, uc = np.genfromtxt(dld.startpath+'/options/populations.options',usecols=(0,1),dtype=str,unpack=True)   
    try:
        ucfloat = uc.astype(float)
        h = ucfloat[setting=="Cosmology_h"][0]
        Omega_m = ucfloat[setting=="Cosmology_Omega_M"][0]
        Omega_k = ucfloat[setting=="Cosmology_Omega_k"][0]
        Omega_L = ucfloat[setting=="Cosmology_Omega_L"][0]
    except TypeError:
        print("Error: Unable to understand cosmology options file. Please check and try again.")
        raise SystemExit
    return h, Omega_m, Omega_k, Omega_L

        
#cosmological definitions. self written is faster than cosmolopy.
#we use these in the quasar class, and the integrals later on
#H(z)
def _integrand(zba, omegam, omegade, omegak):
    return 1.0/np.sqrt((omegam*(1.+zba)**3. + omegade + omegak*(1.+zba)**2.))
#Integrated
def _hubble(z,omegam, omegade, omegak):
    I = quad(_integrand,0., z,args = (omegam,omegade,omegak))
    return I
#dlz; convert to pc
def _dlz(z,omegam,omegade, omegak,h):
    hubbleint = _hubble(z,omegam,omegade,omegak)[0]
    distance = 2997.97*(1+z)* hubbleint/h
    return distance * 1.E6
#comoving volume element dV/dzdA; return in Mpc^3 / sr
def _dvc(z,omegam,omegade,omegak,h):
    hubbleint = _hubble(z,omegam,omegade,omegak)[0]
    comdistance = 2997.97 * hubbleint/h
    vol = 2997.97 * (comdistance**2.) / (h*np.sqrt((omegam*(1.+z)**3. + omegade + omegak*(1.+z)**2.)))
    return vol

#calculate & interpolate cosmology, produce lookup tables of luminosity distance and comoving volume
#for speed we can load up the cosmology lookup tables from a shelf file. 
#this function does this - or creates the shelf in the first place if it does not exist where expected.
#either way it returns our lookup table
#it can store multiple cosmologies.
def _cosmology_lookups():
    _h,_OM,_Ok,_OL = _check_cosmology()
    try:
        s = shelve.open(dld.startpath+"/models/quasars/db/HZQ.db", flag='r')
        _dllookup = s[f'{_h:.3f}_{_OM:.3f}_{_Ok:.3f}_{_OL:.3f}']['dl']
        _dvlookup = s[f'{_h:.3f}_{_OM:.3f}_{_Ok:.3f}_{_OL:.3f}']['dv']
        s.close()
    #normally it is not ideal to not specify the exceptions you want to catch but if s is not present above 
    #I'm not sure how to catch it!!
    except:
        _dllookup = {} 
        _dvlookup = {}
        for z in _myqzs:
            _dllookup[f'{z:.2f}'] = round(_dlz(z,_OM,_OL,_Ok,_h),5)
            _dvlookup[f'{z:.2f}'] = round(_dvc(z,_OM,_OL,_Ok,_h),5)
        try:
            s = shelve.open(dld.startpath+"/models/quasars/db/HZQ.db")
            s[f'{_h:.3f}_{_OM:.3f}_{_Ok:.3f}_{_OL:.3f}'] = {'dl':_dllookup,'dv':_dvlookup}
            s.close()
        except:
            print("Error outputting tables for this cosmology.")
            print("Code will continue but tables will not be saved for future use.")
    return _dllookup,_dvlookup


########################################################################################################################################

#Produce lookup tables for model colours

########################################################################################################################################


#load important details from quasar option file - template distribution, integration limits, and nearzone choice 
def _quasar_options(Jsys):
    #get the settings and (u)ser (c)hoices
    setting,uc = np.genfromtxt(dld.startpath+'/options/populations.options',usecols=(0,1),dtype=str,unpack=True)
    dist_options = {}
    templatetypes=[]
    #check all the values in the options file are floats and add up to the correct amount
    try:
        nz = int(uc[setting=='nz'][0])
        #float version of choices
        ucfloat = uc.astype(float)
        #l/c and lf/cf are the line/continuum options and the chosen fractions
        ml = np.core.defchararray.startswith(setting,'l'); l = setting[ml]; lf = ucfloat[ml]
        mc = np.core.defchararray.startswith(setting,'c'); c = setting[mc]; cf = ucfloat[mc]
        intlo = ucfloat[setting=='zlo'][0]
        inthi = ucfloat[setting=='zhi'][0]
       
        
        #check the line/slope fractions
        if round(np.sum(lf),2) != 1.:
            print(f"Error: Distribution of line widths must sum to 1. Currently they sum to {np.sum(lf)}")
            print("Please check options file and try again.")
            raise SystemExit
        elif round(np.sum(cf),2) != 1.:
            print(f"Error: Distribution of continuum slopes must sum to 1. Currently they sum to {np.sum(cf)}")
            print("Please check options file and try again.")
            raise SystemExit       
        elif intlo >= inthi:
            print("Error: upper integration limit must be higher than lower integration limit.")
            print("Please check options file and try again.")
            raise SystemExit
        #warn the user if integration limits are too high and set defaults.
        if Jsys == 'EUC':
            if intlo > 11.5:                
                print("Error: require integration limits z <= 11.5 for Euclid photometry.")
                print("Setting lower integration limit to z = 7")            
                intlo = 7.
            if inthi > 11.5:
                print("Error: require integration limits z <= 11.5 for Euclid photometry.")
                print("Setting upper integration limit to z = 11.5")
                inthi = 11.5
        elif Jsys == 'MKO':        
            if intlo > 10.:                
                print("Error: require integration limits z <= 10 for MKO photometry.")
                print("Setting lower integration limit to z = 7")            
                intlo = 7.
            if inthi > 10.:
                print("Error: require integration limits z <= 10 for MKO photometry.")
                print("Setting upper integration limit to z = 10")
                inthi = 10.
    except ValueError:
        print("Error: Unable to understand options file. Please check and try again.")
        raise SystemExit    

    for i in range(len(l)):
        for j in range(len(c)):
            tf = round(lf[i] * cf[j],3)
            t = f"{l[i]}_{c[j]}"
            templatetypes.append(t)
            if tf > 0:
                dist_options[t] = {'frac':round(tf,5)}
            
    return dist_options,[intlo,inthi],nz,templatetypes

#set up quasar population templates - colours and k corrections
def _colour_shelf(nz=0):
    print("Loading colours for the high-z quasar population...")
    qset = 'nz3' if nz else 'original'
    try:
        s = shelve.open(dld.startpath+"/models/quasars/db/HZQ.db", flag='r')
        model = s[qset]
        s.close()
    except:
    #if True:
        model = {'l0_cb':{},'l0_cs':{},'l0_cr':{},\
                 'lh_cb':{},'lh_cs':{},'lh_cr':{},\
                 'ls_cb':{},'ls_cs':{},'ls_cr':{},\
                 'ld_cb':{},'ld_cs':{},'ld_cr':{}}
        cols = ['z','z_LSST_MKO','z_PS_MKO','z_DEC_MKO','z_COS_MKO','z_SDSS_MKO','y_LSST_MKO','y_PS_MKO',\
               'Y_DEC_MKO','y_COS_MKO','Z_MKO_MKO','Y_MKO_MKO','J_MKO_MKO','H_MKO_MKO','K_MKO_MKO',\
               'KS_MKO_MKO','W1_WISE_MKO','W2_WISE_MKO','O_EUC_MKO','Y_EUC_MKO','J_EUC_MKO','H_EUC_MKO','KJ_MKO',\
               'z_LSST_EUC','z_PS_EUC','z_DEC_EUC','z_COS_EUC','z_SDSS_EUC','y_LSST_EUC','y_PS_EUC',\
               'Y_DEC_EUC','y_COS_EUC','Z_MKO_EUC','Y_MKO_EUC','J_MKO_EUC','H_MKO_EUC','K_MKO_EUC',\
               'KS_MKO_EUC','W1_WISE_EUC','W2_WISE_EUC','O_EUC_EUC','Y_EUC_EUC','J_EUC_EUC','H_EUC_EUC','KJ_EUC',\
               'i_LSST_MKO','i_PS_MKO','i_DEC_MKO','i_SDSS_MKO','i_COS_MKO',\
               'i_LSST_EUC','i_PS_EUC','i_DEC_EUC','i_SDSS_EUC','i_COS_EUC']
        icols = ['i_SDSS_MKO','i_SDSS_EUC','i_LSST_MKO','i_LSST_EUC','i_PS_MKO','i_PS_EUC']
        for m in model:
        #for m in ['ls_cs']:
            model[m]['X-J_MKO']={}
            model[m]['X-J_EUC']={}
            for c in cols + icols:
                if c.endswith('_MKO'):
                    model[m]['X-J_MKO'][c[:-4]] = {}
                elif c.endswith('_EUC'):
                    model[m]['X-J_EUC'][c[:-4]] = {}
            #we want to produce the X-J colour in all bands X for J_MKO and J_EUC
            #additionally produce K-corrections
            d = pd.read_csv(dld.startpath+f'/models/quasars/X-J/{qset}/colours_kcorr_{m}.ab',\
                            sep='\t',header=64,names=cols,na_values='---')        
            #replace the nas with high value which can be skipped later on in the code
            for col in d:
                ival = d[col].last_valid_index()
                if d[col][ival] > 0:
                    d[col] = d[col].fillna(100)
                else:
                    d[col] = d[col].fillna(-100)
            #interpolate all the colours
            i_PS_m = interpolate.UnivariateSpline(d.z,d.i_PS_MKO,s=0,k=1)            
            z_PS_m = interpolate.UnivariateSpline(d.z,d.z_PS_MKO,s=0,k=1)
            y_PS_m = interpolate.UnivariateSpline(d.z,d.y_PS_MKO,s=0,k=1)
            i_LSST_m = interpolate.UnivariateSpline(d.z,d.i_LSST_MKO,s=0,k=1)            
            z_LSST_m = interpolate.UnivariateSpline(d.z,d.z_LSST_MKO,s=0,k=1)
            y_LSST_m = interpolate.UnivariateSpline(d.z,d.y_LSST_MKO,s=0,k=1)
            i_DEC_m = interpolate.UnivariateSpline(d.z,d.i_DEC_MKO,s=0,k=1)
            z_DEC_m = interpolate.UnivariateSpline(d.z,d.z_DEC_MKO,s=0,k=1)
            Y_DEC_m = interpolate.UnivariateSpline(d.z,d.Y_DEC_MKO,s=0,k=1)
            i_COS_m = interpolate.UnivariateSpline(d.z,d.i_COS_MKO,s=0,k=1)
            z_COS_m = interpolate.UnivariateSpline(d.z,d.z_COS_MKO,s=0,k=1)
            y_COS_m = interpolate.UnivariateSpline(d.z,d.y_COS_MKO,s=0,k=1)
            i_SDSS_m = interpolate.UnivariateSpline(d.z,d.i_SDSS_MKO,s=0,k=1)
            z_SDSS_m = interpolate.UnivariateSpline(d.z,d.z_SDSS_MKO,s=0,k=1)
            Z_MKO_m = interpolate.UnivariateSpline(d.z,d.Z_MKO_MKO,s=0,k=1)
            Y_MKO_m = interpolate.UnivariateSpline(d.z,d.Y_MKO_MKO,s=0,k=1)
            J_MKO_m = interpolate.UnivariateSpline(d.z,d.J_MKO_MKO,s=0,k=1)
            H_MKO_m = interpolate.UnivariateSpline(d.z,d.H_MKO_MKO,s=0,k=1)
            K_MKO_m = interpolate.UnivariateSpline(d.z,d.K_MKO_MKO,s=0,k=1)
            KS_MKO_m = interpolate.UnivariateSpline(d.z,d.KS_MKO_MKO,s=0,k=1)
            W1_m = interpolate.UnivariateSpline(d.z,d.W1_WISE_MKO,s=0,k=1)
            W2_m = interpolate.UnivariateSpline(d.z,d.W2_WISE_MKO,s=0,k=1)
            OE_m = interpolate.UnivariateSpline(d.z,d.O_EUC_MKO,s=0,k=1)
            YE_m = interpolate.UnivariateSpline(d.z,d.Y_EUC_MKO,s=0,k=1)
            JE_m = interpolate.UnivariateSpline(d.z,d.J_EUC_MKO,s=0,k=1)
            HE_m = interpolate.UnivariateSpline(d.z,d.H_EUC_MKO,s=0,k=1)
            KJ_m = interpolate.UnivariateSpline(d.z,d.KJ_MKO,s=0,k=1)
            i_PS_e = interpolate.UnivariateSpline(d.z,d.i_PS_EUC,s=0,k=1)
            z_PS_e = interpolate.UnivariateSpline(d.z,d.z_PS_EUC,s=0,k=1)
            y_PS_e = interpolate.UnivariateSpline(d.z,d.y_PS_EUC,s=0,k=1)
            i_LSST_e = interpolate.UnivariateSpline(d.z,d.i_LSST_EUC,s=0,k=1)
            z_LSST_e = interpolate.UnivariateSpline(d.z,d.z_LSST_EUC,s=0,k=1)
            y_LSST_e = interpolate.UnivariateSpline(d.z,d.y_LSST_EUC,s=0,k=1)
            i_DEC_e = interpolate.UnivariateSpline(d.z,d.i_DEC_EUC,s=0,k=1)
            z_DEC_e = interpolate.UnivariateSpline(d.z,d.z_DEC_EUC,s=0,k=1)
            Y_DEC_e = interpolate.UnivariateSpline(d.z,d.Y_DEC_EUC,s=0,k=1)
            i_COS_e = interpolate.UnivariateSpline(d.z,d.z_COS_EUC,s=0,k=1)
            z_COS_e = interpolate.UnivariateSpline(d.z,d.z_COS_EUC,s=0,k=1)
            y_COS_e = interpolate.UnivariateSpline(d.z,d.y_COS_EUC,s=0,k=1)
            i_SDSS_e = interpolate.UnivariateSpline(d.z,d.z_SDSS_EUC,s=0,k=1)
            z_SDSS_e = interpolate.UnivariateSpline(d.z,d.z_SDSS_EUC,s=0,k=1)
            Z_MKO_e = interpolate.UnivariateSpline(d.z,d.Z_MKO_EUC,s=0,k=1)
            Y_MKO_e = interpolate.UnivariateSpline(d.z,d.Y_MKO_EUC,s=0,k=1)
            J_MKO_e = interpolate.UnivariateSpline(d.z,d.J_MKO_EUC,s=0,k=1)
            H_MKO_e = interpolate.UnivariateSpline(d.z,d.H_MKO_EUC,s=0,k=1)
            K_MKO_e = interpolate.UnivariateSpline(d.z,d.K_MKO_EUC,s=0,k=1)
            KS_MKO_e = interpolate.UnivariateSpline(d.z,d.KS_MKO_EUC,s=0,k=1)
            W1_e = interpolate.UnivariateSpline(d.z,d.W1_WISE_EUC,s=0,k=1)
            W2_e = interpolate.UnivariateSpline(d.z,d.W2_WISE_EUC,s=0,k=1)
            OE_e = interpolate.UnivariateSpline(d.z,d.O_EUC_EUC,s=0,k=1)
            YE_e = interpolate.UnivariateSpline(d.z,d.Y_EUC_EUC,s=0,k=1)
            JE_e = interpolate.UnivariateSpline(d.z,d.J_EUC_EUC,s=0,k=1)
            HE_e = interpolate.UnivariateSpline(d.z,d.H_EUC_EUC,s=0,k=1)
            KJ_e = interpolate.UnivariateSpline(d.z,d.KJ_EUC,s=0,k=1)
            #build the lookup table
            for z in _myqzs:
                model[m]['X-J_MKO']['i_PS'][f"{z:.2f}"] = i_PS_m(z)
                model[m]['X-J_MKO']['z_PS'][f"{z:.2f}"] = z_PS_m(z)
                model[m]['X-J_MKO']['y_PS'][f"{z:.2f}"] = y_PS_m(z)
                model[m]['X-J_MKO']['i_LSST'][f"{z:.2f}"] = i_LSST_m(z)
                model[m]['X-J_MKO']['z_LSST'][f"{z:.2f}"] = z_LSST_m(z)
                model[m]['X-J_MKO']['y_LSST'][f"{z:.2f}"] = y_LSST_m(z)
                model[m]['X-J_MKO']['i_DEC'][f"{z:.2f}"] = i_DEC_m(z)
                model[m]['X-J_MKO']['z_DEC'][f"{z:.2f}"] = z_DEC_m(z)
                model[m]['X-J_MKO']['Y_DEC'][f"{z:.2f}"] = Y_DEC_m(z)
                model[m]['X-J_MKO']['i_COS'][f"{z:.2f}"] = i_COS_m(z)
                model[m]['X-J_MKO']['z_COS'][f"{z:.2f}"] = z_COS_m(z)
                model[m]['X-J_MKO']['y_COS'][f"{z:.2f}"] = y_COS_m(z)
                model[m]['X-J_MKO']['i_SDSS'][f"{z:.2f}"] = i_SDSS_m(z)
                model[m]['X-J_MKO']['z_SDSS'][f"{z:.2f}"] = z_SDSS_m(z)                
                model[m]['X-J_MKO']['Z_MKO'][f"{z:.2f}"] = Z_MKO_m(z)
                model[m]['X-J_MKO']['Y_MKO'][f"{z:.2f}"] = Y_MKO_m(z)
                model[m]['X-J_MKO']['J_MKO'][f"{z:.2f}"] = J_MKO_m(z)
                model[m]['X-J_MKO']['H_MKO'][f"{z:.2f}"] = H_MKO_m(z)
                model[m]['X-J_MKO']['K_MKO'][f"{z:.2f}"] = K_MKO_m(z)
                model[m]['X-J_MKO']['KS_MKO'][f"{z:.2f}"] = KS_MKO_m(z)
                model[m]['X-J_MKO']['W1_WISE'][f"{z:.2f}"] = W1_m(z)
                model[m]['X-J_MKO']['W2_WISE'][f"{z:.2f}"] = W2_m(z)
                model[m]['X-J_MKO']['O_EUC'][f"{z:.2f}"] = OE_m(z)
                model[m]['X-J_MKO']['Y_EUC'][f"{z:.2f}"] = YE_m(z)
                model[m]['X-J_MKO']['J_EUC'][f"{z:.2f}"] = JE_m(z)
                model[m]['X-J_MKO']['H_EUC'][f"{z:.2f}"] = HE_m(z)
                model[m]['X-J_MKO']['KJ'][f"{z:.2f}"] = KJ_m(z)
                model[m]['X-J_EUC']['i_PS'][f"{z:.2f}"] = i_PS_e(z)
                model[m]['X-J_EUC']['z_PS'][f"{z:.2f}"] = z_PS_e(z)
                model[m]['X-J_EUC']['y_PS'][f"{z:.2f}"] = y_PS_e(z)
                model[m]['X-J_EUC']['i_LSST'][f"{z:.2f}"] = i_LSST_e(z)
                model[m]['X-J_EUC']['z_LSST'][f"{z:.2f}"] = z_LSST_e(z)
                model[m]['X-J_EUC']['y_LSST'][f"{z:.2f}"] = y_LSST_e(z)
                model[m]['X-J_EUC']['i_DEC'][f"{z:.2f}"] = i_DEC_e(z)
                model[m]['X-J_EUC']['z_DEC'][f"{z:.2f}"] = z_DEC_e(z)
                model[m]['X-J_EUC']['Y_DEC'][f"{z:.2f}"] = Y_DEC_e(z)
                model[m]['X-J_EUC']['i_COS'][f"{z:.2f}"] = i_COS_e(z)
                model[m]['X-J_EUC']['z_COS'][f"{z:.2f}"] = z_COS_e(z)
                model[m]['X-J_EUC']['y_COS'][f"{z:.2f}"] = y_COS_e(z) 
                model[m]['X-J_EUC']['i_SDSS'][f"{z:.2f}"] = i_SDSS_e(z)
                model[m]['X-J_EUC']['z_SDSS'][f"{z:.2f}"] = z_SDSS_e(z)                
                model[m]['X-J_EUC']['Z_MKO'][f"{z:.2f}"] = Z_MKO_e(z)
                model[m]['X-J_EUC']['Y_MKO'][f"{z:.2f}"] = Y_MKO_e(z)
                model[m]['X-J_EUC']['J_MKO'][f"{z:.2f}"] = J_MKO_e(z)
                model[m]['X-J_EUC']['H_MKO'][f"{z:.2f}"] = H_MKO_e(z)
                model[m]['X-J_EUC']['K_MKO'][f"{z:.2f}"] = K_MKO_e(z)
                model[m]['X-J_EUC']['KS_MKO'][f"{z:.2f}"] = KS_MKO_e(z)
                model[m]['X-J_EUC']['W1_WISE'][f"{z:.2f}"] = W1_e(z)
                model[m]['X-J_EUC']['W2_WISE'][f"{z:.2f}"] = W2_e(z)
                model[m]['X-J_EUC']['O_EUC'][f"{z:.2f}"] = OE_e(z)
                model[m]['X-J_EUC']['Y_EUC'][f"{z:.2f}"] = YE_e(z)
                model[m]['X-J_EUC']['J_EUC'][f"{z:.2f}"] = JE_e(z)
                model[m]['X-J_EUC']['H_EUC'][f"{z:.2f}"] = HE_e(z)
                model[m]['X-J_EUC']['KJ'][f"{z:.2f}"] = KJ_e(z)
                #do the ugr bands which we always just want at a high value
                #not all of these systems have the full set ugr but won't matter that extras are created
                for opt_band in 'rgu':
                    for opt_set in ['PS','LSST','SDSS','DEC','COS']:
                        try:
                            model[m]['X-J_MKO'][f'{opt_band}_{opt_set}'][f"{z:.2f}"] = 100
                            model[m]['X-J_EUC'][f'{opt_band}_{opt_set}'][f"{z:.2f}"] = 100
                        except KeyError:
                            model[m]['X-J_MKO'][f'{opt_band}_{opt_set}'] = {f"{z:.2f}": 100}
                            model[m]['X-J_EUC'][f'{opt_band}_{opt_set}'] = {f"{z:.2f}": 100}
        try:
            s = shelve.open(dld.startpath+"/models/quasars/db/HZQ.db",'n')
            s[qset] = model
            s.close()
        except:
            print("Error saving HZQ colour table.")
            print("Code will continue but colours will not be saved for future use.")
    return model

########################################################################################################################################

#Define functions required for BMC calc

########################################################################################################################################

#absolute magnitude limits required for integrals
def _Mlimits(Jflx,Jerr,zin,qm,Jfilter,nsig):
    #dlz is returned in pc
    d_l = _dllookup[f"{zin:.2f}"]
    Jup = -2.5*np.log10((Jflx + (nsig*Jerr))/3631.E0)
    #slight fudge to prevent an error if J-detection is below 5-sigma
    Jlo = -2.5*np.log10((Jflx - (nsig*Jerr))/3631.E0) if Jflx - (nsig*Jerr) > 0. else -2.5*np.log10((0.1*Jerr)/3631.E0)
    Mmin = round(Jup - ((5.*np.log10(d_l))-5.) + (2.5*np.log10(1.+zin)) + _model[qm][f'X-J_{Jfilter}']['KJ'][f"{zin:.2f}"],4)
    Mmax = round(Jlo - ((5.*np.log10(d_l))-5.) + (2.5*np.log10(1.+zin)) + _model[qm][f'X-J_{Jfilter}']['KJ'][f"{zin:.2f}"],4)
    return [Mmin,Mmax]

#surface density of quasars (per sr) using the Jiang+2016 QLF
def _qsodensity(M,zin):
    numerator = (9.93E-9)*(10.**(-0.70*(zin - 6.0)))
    denominator = (10.**(-0.36*(M+25.2))) + (10.**(-0.72*(M+25.2)))
    qlf = numerator/denominator
    #note that this element is dV/dzdA; i.e. per unit area in sr as required.
    dVc = _dvlookup[f"{zin:.2f}"]
    return dVc*qlf

#this single power law version was used for the VIKING search and could be reintroduced if desired
#you get a systematic shift of the quasar weights depending on what QLF is used.
#in _qsointegrand() change the prior function that is used in line 347
def _qsodensitysingle(M,zin):
    qlf = (5.2E-9)*(10.**((0.84*(M+26.0)) - (0.47*(zin-6.0))))
    dVc = _dvlookup[f"{zin:.2f}"]
    return dVc*qlf

def _Jflx_from_M1450(M,zin,qm,Jfilter):
    d_l = _dllookup[f"{zin:.2f}"]
    #always start with a model J band flux based on the value of M passed in
    return 3631.*(10**(-0.4*(M+(5.*np.log10(d_l))-5.-(2.5*np.log10(1.+zin))-_model[qm][f'X-J_{Jfilter}']['KJ'][f"{zin:.2f}"])))

#by this stage we expect photometry in linear fluxes, or upper limits
#this will be stored in a dictionary with labels indicating the filters
#this is why everything is linked to the J bands.
def _qsolikelihood(M,zin,qm,Jfilter,photom):
    #always start with a model J band flux based on the value of M passed in
    predJ = _Jflx_from_M1450(M,zin,qm,Jfilter)
    lk = 1.
    for band in photom:
        if photom[band]['f'] is not None:
            colour = _model[qm][f'X-J_{Jfilter}'][bmccom.band_name_nodigits_regex(band)][f"{zin:.2f}"]
        #the code will have already checked the redshift limits
        #so very blue H/K/W-J colours (z>10) are not an issue (where template fJ = 0).
        #we now deal with very red X-J colours in bluer X bands,
        #where template flux can be zero.
            predb = predJ * 10**(-0.4*colour) if colour < 8 else 0.
            lk *= bmccom.liketerm(predb,photom[band]['f'],photom[band]['e'])
    return lk

def _qsointegrand(M,zin,qm,Jfilter,photom):
    return _qsodensity(M,zin)*_qsolikelihood(M,zin,qm,Jfilter,photom)

def W(phot,Jfilter):
    Jfilter = Jfilter[:3]#the filter may have VIK in the name which only matters for ETGs
    fJ,eJ = phot[f"J_{Jfilter}1"]['f'],phot[f"J_{Jfilter}1"]['e']
    Wq = 0.
    for m in _qtempdist:
        #integral evalauted over M at points in z
        intM_z = []
        for z in _simpz_q:
            Mlims = _Mlimits(fJ,eJ,z,m,Jfilter,5.)
            intM_z.append(quad(_qsointegrand,Mlims[0],Mlims[1],args=(z,m,Jfilter,phot))[0])
        Wq += _qtempdist[m]['frac'] * bmccom.simpson(_simpz_q,intM_z)
    return Wq

#return quasar weight as function of z (i.e. total W above = integral dW_dz dz)
def dW_dz(phot,Jfilter):
    Jfilter = Jfilter[:3]#the filter may have VIK in the name which only matters for ETGs
    fJ,eJ = phot[f"J_{Jfilter}1"]['f'],phot[f"J_{Jfilter}1"]['e']
    dWdz = []
    for z in _simpz_q:
        ztot = 0.
        for m in _qtempdist:
            Mlims = _Mlimits(fJ,eJ,z,m,Jfilter,5.)
            intJ = _qtempdist[m]['frac'] * quad(_qsointegrand,Mlims[0],Mlims[1],args=(z,m,Jfilter,phot))[0]
            ztot += intJ
        dWdz.append(ztot)
    return {'z':_simpz_q, 'vals':dWdz}


#return best fitting SED
#only fit to fluxes - skip filters with limits.
def best_fit_SED(phot,Jfilter):
    Jfilter = Jfilter[:3]#the filter may have VIK in the name which only matters for ETGs prior
    bands,sed,errors = bmccom.seds_for_chisq(phot,Jfilter)
    Jflx = phot[f"J_{Jfilter}1"]['f']#scale all templates to measured J
    chisqmin = -100
    for z in _simpz_q: #iterate over models
        for m in _allqms:
            template = []
            for band in bands:#build template for each model. note the bands in 'bands' have had subscripted integers removed.
                colour = _model[m][f'X-J_{Jfilter}'][band][f"{z:.2f}"]
                predb = Jflx * 10**(-0.4*colour) if colour < 8 else 0.
                template.append(predb)
            best_scale = bmccom.scalebest(template,sed,errors)#scaling that minimises chisq for a template; scalebest also excludes limits
            chisq = sum([bmccom.chisq(template[i],j,errors[i],best_scale) for i,j in enumerate(sed) if type(errors[i])!=str])#minimum chisq, exclude limits
            if chisqmin == -100 or chisq < chisqmin: #keep template details if it is the best fit so far
                chisqmin = chisq
                best_template = template
                best_sf = best_scale
                best_m = m
                best_z = z
    return {'bands':bands,'f':sed,'e':errors,'template':best_template,'scaling':best_sf,'qm':best_m,'qz':best_z,'chisq':chisqmin, 'wavelengths':[dld._eff_wavelength[b] for b in bands]}


if __name__ == "__main__":
    print("Value of __name__ is:", __name__)
    print("Running hzq.py module")
    print(_check_cosmology())
        
    print('done - hzqs')
    
    #do whatever
else:
    #myqzs is used to evaluate cosmology at required points
#which we output to lookup table.
#they are also the evaluation points for interpolated colours, for the lookup table
#the integral is carried out at dz = 0.05 so this spacing is ample.
    _myqzs = np.arange(5.2,12.0001,0.01)
    _dllookup,_dvlookup = _cosmology_lookups()
    _qtempdist,_qint_lims,_nz,_allqms = _quasar_options(dld._Jfilt)
    _simpz_q = np.arange(_qint_lims[0],_qint_lims[1]+0.01,0.05)
    _model = _colour_shelf(_nz)

