#python 3
#this script is part of the Pq server code
#it sets up the MLT dwarf population
import numpy as np
from scipy.integrate import quad
import shelve
#custom modules
import bmc_common as bmccom
import dataload as dld

########################################################################################################################################

#Produce lookup tables for model colours

########################################################################################################################################

def _mlt_options():
    #get the settings and (u)ser (c)hoices
    setting,uc = np.genfromtxt(dld.startpath+'/options/populations.options',usecols=(0,1),dtype=str,unpack=True)
    try:
        #float version of choices
        ucfloat = uc.astype(float)
        #l/c and lf/cf are the line/continuum options and the chosen fractions
        mlt = np.core.defchararray.startswith(setting,'MLT_scatter'); 
        return ucfloat[mlt][0]
    except ValueError:
        print("Error: Unable to understand options file. Please check and try again.")
        raise SystemExit


#for speed we can load up the MLT lookup table from a shelf file. 
#this function does this - or creates the shelf in the first place if it does not exist where expected.
#either way it returns our lookup table

#all objects in the lookup table are the band - J colour, for Jmko and JEUC. 
#this will let us select model colours easily when the bands for a given set are known.
def _colour_shelf():
    print("Loading colours for the MLT population...")
    #if the code has run before we can just extract the required lookup table directly
    try:
        s = shelve.open(dld.startpath+"/models/MLT/db/MLT.db", flag='r')
        model = s['MLT']
        s.close()
    #if not we can create the lookup from scratch   
    #normally it is not ideal to not specify the exceptions you want to catch but if s is not present above 
    #I'm not sure how to catch it!!
    except:
        model = {}
        with open(dld.startpath+'/models/MLT/MLTcolours.ab', 'r') as _ofile:
            _data = _ofile.readlines()[34:]
            for _row in _data:
                _cols = _row.split()
                _t = _cols[0]
                #measured colours from the input file.
                #we ombine these to make X-J colours in every band X.
                #this makes it easier to flexibly select filters later on.
                #no need to store the ascii colour now. 
                iz_PS = float(_cols[4]); iz_SDSS = float(_cols[5]); zy_PS = float(_cols[6]);\
zY_PS_MKO= float(_cols[7]); zY_DEC= float(_cols[8]); zY_DEC_MKO= float(_cols[9]);\
zY_COS_MKO= float(_cols[10]);zY_SDSS_MKO= float(_cols[11]); ZY_MKO= float(_cols[12]);\
YJ_MKO= float(_cols[13]);JH_MKO= float(_cols[14]);HK_MKO= float(_cols[15]);KKs_MKO= float(_cols[16]);\
KW1= float(_cols[17]); W1W2= float(_cols[18]);\
OY_EUC= float(_cols[19]);YJ_EUC= float(_cols[20]);JH_EUC= float(_cols[21]);JJ_EUC_MKO= float(_cols[22])
                #first objects for the lookup
                model[_t] = {'n':float(_cols[1]), 'MJ_MKO': float(_cols[2]), 'MJ_EUC': float(_cols[3])}
                #now combine and store all the colours
                model[_t]['X-J_MKO']=\
                {'i_PS': iz_PS + zY_PS_MKO + YJ_MKO,\
                 'z_PS': zY_PS_MKO + YJ_MKO,\
                 'y_PS': zY_PS_MKO + YJ_MKO - zy_PS,\
                 #note LSST & PS have same colours
                 'i_LSST': iz_PS + zY_PS_MKO + YJ_MKO,\
                 'z_LSST': zY_PS_MKO + YJ_MKO,\
                 'y_LSST': zY_PS_MKO + YJ_MKO - zy_PS,\
                 'i_SDSS': iz_SDSS + zY_SDSS_MKO + YJ_MKO,\
                 'z_SDSS': zY_SDSS_MKO + YJ_MKO,\
                 'z_DEC': zY_DEC_MKO + YJ_MKO,\
                 'Y_DEC': zY_DEC_MKO + YJ_MKO - zY_DEC,\
                 #zy DEC matches zy COSMOS but the MKO links are different :S
                 'z_COS': zY_COS_MKO + YJ_MKO,\
                 'y_COS': zY_COS_MKO + YJ_MKO - zY_DEC,\
                 'Z_MKO': ZY_MKO + YJ_MKO,\
                 'Y_MKO': YJ_MKO,\
                 'J_MKO': 0.,\
                 'H_MKO': -JH_MKO,\
                 'K_MKO': -(JH_MKO + HK_MKO),\
                 'KS_MKO':-(JH_MKO + HK_MKO + KKs_MKO),\
                 'W1_WISE':-(JH_MKO + HK_MKO + KW1),\
                 'W2_WISE':-(JH_MKO + HK_MKO + KW1 + W1W2),\
                 'O_EUC': OY_EUC + YJ_EUC + JJ_EUC_MKO,\
                 'Y_EUC': YJ_EUC + JJ_EUC_MKO,\
                 'J_EUC': JJ_EUC_MKO,\
                 'H_EUC': JJ_EUC_MKO - JH_EUC}
                #now we have a shortcut to the X-J_EUC colours 
                model[_t]['X-J_EUC']={}
                for X in model[_t]['X-J_MKO']:
                    model[_t]['X-J_EUC'][X] = model[_t]['X-J_MKO'][X] - JJ_EUC_MKO                    
        s = shelve.open(dld.startpath+"/models/MLT/db/MLT.db",'n')
        s['MLT'] = model
        s.close()
    return model


########################################################################################################################################

#Define functions required for BMC calc

########################################################################################################################################

#integration limits
def _fjlimits(bdfJ,bdeJ,nsig):
    Jup = bdfJ + (nsig*bdeJ)
    Jlo = bdfJ - (nsig*bdeJ) if bdfJ - (nsig*bdeJ) > 0. else 0.1*bdeJ
    return[Jlo,Jup]
#need d in pc - this is needed for the scale height    
def _dfromfj(fj,t,Jfilter):
    jmag = -2.5*np.log10(fj/3631.)
    return 10**(0.2*(jmag-_model[t][f'MJ_{Jfilter}'] + 5))

#surface density of sources
#note that we originally derived dN/dJ and have included a dJ/df factor (2.5/f*ln10)
#for our final integral which is in flux space
#this should represent dN/(dfdA) with A area in steradians
def _mltdensity(fj,t,sb,Jfilter):
    exponent = -1.5 * np.log10(fj/(3631.))
    d_fj = _dfromfj(fj,t,Jfilter)
    H = d_fj * sb
    return 0.5*_model[t]['n']*np.exp(-1.0*H/300.)*(10**(0.6*(5-_model[t][f'MJ_{Jfilter}'])))*(10**exponent)/fj
#fluxes have gaussian error distribution
#need flexible choice of optical band
def _mltlikelihood(fj,t,Jfilter,photom):
    lk = 1.
    for band in photom:
        if photom[band]['f']:
            colour = _model[t][f'X-J_{Jfilter}'][band[:-1]]
            predb = fj * 10**(-0.4*colour)
            lk *= bmccom.liketerm(predb,photom[band]['f'],photom[band]['e_mlt'])
    return lk

def _mltintegrand(fj,t,sb,Jfilter,phot):
    return _mltdensity(fj,t,sb,Jfilter)*_mltlikelihood(fj,t,Jfilter,phot)

#use the stated additional magnitude error to amend the flux errors for the MLT weight calculation
def _add_error_in_quadrature(f,sigf,del_sigm):
     #skip upper limits
    if type(sigf) == str:
        return sigf
    sigm = abs((2.5*sigf)/(2.3*f))
    sigm_plus = np.sqrt(sigm**2. + del_sigm**2)
    return f * sigm_plus * 2.3/2.5

def W(phot,sb,Jfilter):
    Jfilter = Jfilter[:3]#the filter may have VIK in the name which only matters for ETGs
    Ws = 0.
    fJ,eJ = phot[f"J_{Jfilter}1"]['f'],phot[f"J_{Jfilter}1"]['e']
    slims = _fjlimits(fJ,eJ,5.)
    #adjust the flux errors by adding magnitude error in quadrature
    for b in phot:
        if phot[b]['f']:
            phot[b]['e_mlt'] = _add_error_in_quadrature(phot[b]['f'],phot[b]['e'],_sigm_in_quad)
    for t in _spectraltypes:
        Ws += quad(_mltintegrand,slims[0],slims[1], args=(t,sb,Jfilter,phot))[0]
    return Ws

#return the unsummed weights for each spectral type
def dW_dt(phot,sb,Jfilter):
    Jfilter = Jfilter[:3]
    fJ,eJ = phot[f"J_{Jfilter}1"]['f'],phot[f"J_{Jfilter}1"]['e']
    slims = _fjlimits(fJ,eJ,5.)
    #adjust the flux errors by adding magnitude error in quadrature
    for b in phot:
        if phot[b]['f']:
            phot[b]['e_mlt'] = _add_error_in_quadrature(phot[b]['f'],phot[b]['e'],_sigm_in_quad)
    dWdt = [quad(_mltintegrand,slims[0],slims[1], args=(t,sb,Jfilter,phot))[0] for t in _spectraltypes]
    return {'t':_spectraltypes, 'i':np.arange(-0.5,len(_spectraltypes)+0.5,1.), 'vals':dWdt}

#return best fitting SED
#only fit to fluxes - skip filters with limits.
def best_fit_SED(phot,Jfilter):
    Jfilter = Jfilter[:3]#the filter may have VIK in the name which only matters for ETGs prior
    bands,sed,errors = bmccom.seds_for_chisq(phot,Jfilter)
    Jflx = phot[f"J_{Jfilter}1"]['f']#scale all templates to measured J
    chisqmin = -100
    for t in _spectraltypes: #iterate over models
        template = []
        for band in bands:#build template for each model. note the bands in 'bands' have had subscripted integers removed.
            colour = _model[t][f'X-J_{Jfilter}'][band]
            predb = Jflx * 10**(-0.4*colour)
            template.append(predb)
        best_scale = bmccom.scalebest(template,sed,errors)#scaling that minimises chisq for a template
        chisq = sum([bmccom.chisq(template[i],j,errors[i],best_scale) for i,j in enumerate(sed) if type(errors[i])!=str])#minimum chisq
        if chisqmin == -100 or chisq < chisqmin: #keep template details if it is the best fit so far
            chisqmin = chisq
            best_template = template
            best_sf = best_scale
            best_t = t
    return {'bands':bands,'f':sed,'e':errors,'template':best_template,'scaling':best_sf,'spt':best_t,'chisq':chisqmin,'wavelengths':[dld._eff_wavelength[b] for b in bands]}

if __name__ == "__main__":
    print("Value of __name__ is:", __name__)
    print("Running mlt.py module")
    print(_mlt_options())
    #do whatever
else:
    _spectraltypes = ['M0','M1','M2','M3','M4','M5','M6','M7','M8','M9','L0','L1','L2','L3','L4','L5','L6','L7','L8','L9','T0','T1','T2','T3','T4','T5','T6','T7','T8']
    _model = _colour_shelf()
    _sigm_in_quad = _mlt_options()

