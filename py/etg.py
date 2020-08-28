#python 3
#this script is part of the Pq server code
#it sets up the early-type galaxy population
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import shelve
import pandas as pd
#custom modules
import bmc_common as bmccom
import dataload as dld

#for speed we can load up the ETG lookup table from a shelf file. 
#this function does this - or creates the shelf in the first place if it does not exist where expected.
#either way it returns our lookup table

#all objects in the lookup table are the band - J colour, for Jmko and JEUC. 
#this will let us select model colours easily when the bands for a given set are known.
def _colour_shelf():
    print("Loading colours for the early-type galaxy population...")
    #if the code has run before we can just extract the required lookup table directly
    try:
        s = shelve.open(dld.startpath+"/models/galaxies/db/ETG.db", flag='r')
        model = s['ETG']
        s.close()
    #if not we can create the lookup from scratch   
    #normally it is not ideal to not specify the exceptions you want to catch but if s is not present above 
    #I'm not sure how to catch it!! shelf does not seem to raise a normal error type
    except:
        model = {'zf3':{},'zf10':{}}
        cols = ['z','gr_PS','gr_DEC','gr_SDSS','ri_PS','ri_DEC','ri_SDSS','ri_COS',\
                'iz_PS','iz_DEC','iz_SDSS','iz_COS','zy_PS','zY_PS_MKO','zY_DEC','zY_DEC_MKO','zY_SDSS_MKO',\
                'ZY_MKO','YJ_MKO','JH_MKO','HK_MKO','KKs_MKO','KW1','W1W2','OY_EUC','YJ_EUC','JH_EUC','JJ_EUC_MKO',\
                'ug_LSST','ug_SDSS']
        for zf in model:
            interps = {}
            model[zf]['X-J_MKO'] = {}; model[zf]['X-J_EUC'] = {}
            d = pd.read_csv(dld.startpath+f'/models/galaxies/colours_{zf}.ab',sep='\t',header=35,names=cols)
            #recast the colours as X-J for all bands, and J_MKO
            #these bands match the MLT filters so all are available for the BMC
            interps['g_PS'] = interpolate.UnivariateSpline(d.z,d.gr_PS + d.ri_PS + d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['r_PS'] = interpolate.UnivariateSpline(d.z,d.ri_PS + d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['i_PS'] = interpolate.UnivariateSpline(d.z,d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['z_PS'] = interpolate.UnivariateSpline(d.z,d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['y_PS'] = interpolate.UnivariateSpline(d.z,d.zY_PS_MKO + d.YJ_MKO - d.zy_PS,s=0,k=1)
            #note LSST & PS have same colours
            interps['u_LSST'] = interpolate.UnivariateSpline(d.z,d.ug_LSST + d.gr_PS + d.ri_PS + d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['g_LSST'] = interpolate.UnivariateSpline(d.z,d.gr_PS + d.ri_PS + d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['r_LSST'] = interpolate.UnivariateSpline(d.z,d.ri_PS + d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)          
            interps['i_LSST'] = interpolate.UnivariateSpline(d.z,d.iz_PS + d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['z_LSST'] = interpolate.UnivariateSpline(d.z,d.zY_PS_MKO + d.YJ_MKO,s=0,k=1)
            interps['y_LSST'] = interpolate.UnivariateSpline(d.z,d.zY_PS_MKO + d.YJ_MKO - d.zy_PS,s=0,k=1)
            interps['u_SDSS'] = interpolate.UnivariateSpline(d.z,d.ug_SDSS + d.gr_SDSS + d.ri_SDSS + d.iz_SDSS + d.zY_SDSS_MKO + d.YJ_MKO,s=0,k=1)
            interps['g_SDSS'] = interpolate.UnivariateSpline(d.z,d.gr_SDSS + d.ri_SDSS + d.iz_SDSS + d.zY_SDSS_MKO + d.YJ_MKO,s=0,k=1)
            interps['r_SDSS'] = interpolate.UnivariateSpline(d.z,d.ri_SDSS + d.iz_SDSS + d.zY_SDSS_MKO + d.YJ_MKO,s=0,k=1)
            interps['i_SDSS'] = interpolate.UnivariateSpline(d.z,d.iz_SDSS + d.zY_SDSS_MKO + d.YJ_MKO,s=0,k=1)
            interps['z_SDSS'] = interpolate.UnivariateSpline(d.z,d.zY_SDSS_MKO + d.YJ_MKO,s=0,k=1)
            interps['g_DEC'] = interpolate.UnivariateSpline(d.z,d.gr_DEC + d.ri_DEC + d.iz_DEC + d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['r_DEC'] = interpolate.UnivariateSpline(d.z,d.ri_DEC + d.iz_DEC + d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['i_DEC'] = interpolate.UnivariateSpline(d.z,d.iz_DEC + d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['z_DEC'] = interpolate.UnivariateSpline(d.z,d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['Y_DEC'] = interpolate.UnivariateSpline(d.z,d.zY_DEC_MKO + d.YJ_MKO - d.zY_DEC,s=0,k=1)
            #zy DEC matches zy COSMOS but the MKO links are different :S
            interps['r_COS'] = interpolate.UnivariateSpline(d.z,d.ri_COS + d.iz_COS + d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['i_COS'] = interpolate.UnivariateSpline(d.z,d.iz_COS + d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['z_COS'] = interpolate.UnivariateSpline(d.z,d.zY_DEC_MKO + d.YJ_MKO,s=0,k=1)
            interps['y_COS'] = interpolate.UnivariateSpline(d.z,d.zY_DEC_MKO + d.YJ_MKO - d.zY_DEC,s=0,k=1)
            interps['Z_MKO'] = interpolate.UnivariateSpline(d.z,d.ZY_MKO + d.YJ_MKO,s=0,k=1)
            interps['Y_MKO'] = interpolate.UnivariateSpline(d.z,d.YJ_MKO,s=0,k=1)
            interps['J_MKO'] = interpolate.UnivariateSpline(d.z,np.zeros(len(d.z)),s=0,k=1)
            interps['H_MKO'] = interpolate.UnivariateSpline(d.z,-1.0*d.JH_MKO,s=0,k=1)
            interps['K_MKO'] = interpolate.UnivariateSpline(d.z,-1.0*(d.JH_MKO + d.HK_MKO),s=0,k=1)
            interps['KS_MKO'] = interpolate.UnivariateSpline(d.z,-1.0*(d.JH_MKO + d.HK_MKO + d.KKs_MKO),s=0,k=1)
            interps['W1_WISE'] = interpolate.UnivariateSpline(d.z,-1.0*(d.JH_MKO + d.HK_MKO + d.KW1),s=0,k=1)
            interps['W2_WISE'] = interpolate.UnivariateSpline(d.z,-1.0*(d.JH_MKO + d.HK_MKO + d.KW1 + d.W1W2),s=0,k=1)
            interps['O_EUC'] = interpolate.UnivariateSpline(d.z,d.OY_EUC + d.YJ_EUC + d.JJ_EUC_MKO,s=0,k=1)
            interps['Y_EUC'] = interpolate.UnivariateSpline(d.z,d.YJ_EUC + d.JJ_EUC_MKO,s=0,k=1)
            interps['J_EUC'] = interpolate.UnivariateSpline(d.z,d.JJ_EUC_MKO,s=0,k=1)
            interps['H_EUC'] = interpolate.UnivariateSpline(d.z,d.JJ_EUC_MKO - d.JH_EUC ,s=0,k=1)
            for i in interps:
                model[zf]['X-J_MKO'][i] = {}; model[zf]['X-J_EUC'][i] = {}
                for z in _mygzs:
                    model[zf]['X-J_MKO'][i][f"{z:.2f}"] = interps[i](z)
                    model[zf]['X-J_EUC'][i][f"{z:.2f}"] = interps[i](z) - interps['J_EUC'](z)
        try:
            s = shelve.open(dld.startpath+"/models/galaxies/db/ETG.db", flag='n')
            s['ETG'] = model
            s.close()
        except:
            print("Error saving ETG colour table.")
            print("Code will continue but colours will not be saved for future use.")
    return model

########################################################################################################################################

#Define functions required for BMC calc

########################################################################################################################################

def _Jlimits(Jflx,Jerr,nsig):
    Jlo = -2.5*np.log10((Jflx+(nsig*Jerr))/3631.)
    Jup = -2.5*np.log10((Jflx-(nsig*Jerr))/3631.) if Jflx-(5.*Jerr)>0. else -2.5*np.log10((0.1*Jerr)/3631.)
    return[Jlo,Jup]

#specifically for VIKING, we determined an extra function to account for object morphology as a function of S/N in J    
def _mcsfunc(snJ):
    return (4.83691496 * np.exp(-0.5438693 * snJ))

#new galaxy luminosity function; either for VIKING or Euclid magnitudes
def _etgdensity(jab,zin,Jfilter,eJ):
    fJ = 3631.*(10**(-0.4*jab))
    #version suitable for Euclid 1" mags
    if Jfilter[:3] == "EUC":
        fz = 20.692379716597092 + (1.3318688444398394*zin)
        jexponent = -0.5 * (((jab-fz)/0.7701690148332615)**2.)
        zexponent = -1.0 * ((zin-0.8)/0.42356754534101404)
        return (np.exp(jexponent)*np.exp(zexponent)*12377.253038255019*(180./3.14159)**2)/1.38
    #version for MKO filter based on VIKING 2" mags (with COSMOS correction)
    #also includes MCS factor according to whether VIKING data is being used
    elif Jfilter[:3] == "MKO":
        if Jfilter.endswith("VIK"):
            mcsfactor = _mcsfunc(fJ/eJ)
        else:
            mcsfactor = 1.
        jcorr = jab + 0.2
        fz = 20.4665043557 + (1.46181719*zin)
        jexponent = -0.5 * (((jcorr-fz)/0.883057857)**2.)
        zexponent = -1.0 * ((zin-0.8)/0.42864968)
        return mcsfactor * (np.exp(jexponent)*np.exp(zexponent)*10621.7070253*(180./3.14159)**2)/1.38


#by this stage we expect photometry in linear fluxes, or upper limits
#this will be stored in a dictionary with labels indicating the filters
#this is why everything is linked to the J bands.
def _etglikelihood(jab,zin,zf,Jfilter,photom):
    predJ = 3631.*(10**(-0.4*jab))
    lk = 1.
    for band in photom:
        if photom[band]['f'] is not None:
            colour = _model[zf][f'X-J_{Jfilter}'][bmccom.band_name_nodigits_regex(band)][f"{zin:.2f}"]
            predb = predJ * 10**(-0.4*colour)
            lk *= bmccom.liketerm(predb,photom[band]['f'],photom[band]['e'])
    return lk

def _etgintegrand(jab,zin,zf,Jfilter,photom):
    return _etgdensity(jab,zin,Jfilter,photom[f"J_{Jfilter[:3]}1"]['e'])*_etglikelihood(jab,zin,zf,Jfilter[:3],photom)

def W(phot,Jfilter):
    fJ,eJ = phot[f"J_{Jfilter[:3]}1"]['f'],phot[f"J_{Jfilter[:3]}1"]['e']
    glims = _Jlimits(fJ,eJ,5.)
    #integral evalauted over J at points in z for both formation redshifts
    intJ_z3,intJ_z10 = [],[]
    for z in _simpz_g:
        intJ_z3.append(quad(_etgintegrand,glims[0],glims[1],args=(z,'zf3',Jfilter,phot))[0])
        intJ_z10.append(quad(_etgintegrand,glims[0],glims[1],args=(z,'zf10',Jfilter,phot))[0])
    return (0.8*bmccom.simpson(_simpz_g,intJ_z3)) + (0.2*bmccom.simpson(_simpz_g,intJ_z10))

def dW_dz(phot,Jfilter):
    fJ,eJ = phot[f"J_{Jfilter[:3]}1"]['f'],phot[f"J_{Jfilter[:3]}1"]['e']
    glims = _Jlimits(fJ,eJ,5.)
    #integral evalauted over J at points in z for both formation redshifts
    intJ= []
    for z in _simpz_g:
        intJ_z3 = quad(_etgintegrand,glims[0],glims[1],args=(z,'zf3',Jfilter,phot))[0]
        intJ_z10 = quad(_etgintegrand,glims[0],glims[1],args=(z,'zf10',Jfilter,phot))[0]
        intJ.append((0.8*intJ_z3)+(0.2*intJ_z10))
    return {'vals':intJ, 'z':_simpz_g}

#return best fitting SED
#only fit to fluxes - skip filters with limits.
def best_fit_SED(phot,Jfilter):
    Jfilter = Jfilter[:3]#the filter may have VIK in the name which only matters for ETGs prior
    bands,sed,errors = bmccom.seds_for_chisq(phot,Jfilter)
    Jflx = phot[f"J_{Jfilter}1"]['f']#scale all templates to measured J
    chisqmin = -100
    for z in _simpz_g: #iterate over models
        for zf in ['zf3','zf10']:
            template = []
            for band in bands:#build template for each model. note the bands in 'bands' have had subscripted integers removed.
                colour = _model[zf][f'X-J_{Jfilter}'][band][f"{z:.2f}"]
                predb = Jflx * 10**(-0.4*colour)
                template.append(predb)
            best_scale = bmccom.scalebest(template,sed,errors)#scaling that minimises chisq for a template
            chisq = sum([bmccom.chisq(template[i],j,errors[i],best_scale) for i,j in enumerate(sed) if type(errors[i])!=str])#minimum chisq
            if chisqmin == -100 or chisq < chisqmin: #keep template details if it is the best fit so far
                chisqmin = chisq
                best_template = template
                best_sf = best_scale
                best_zf = zf
                best_z = z
    return {'bands':bands,'f':sed,'e':errors,'template':best_template,'scaling':best_sf,'zf':best_zf,'z':best_z,'chisq':chisqmin,'wavelengths':[dld._eff_wavelength[b] for b in bands]}

if __name__ == "__main__":
    print("Value of __name__ is:", __name__)
    print("Running etg.py module")
    print("done - ETG colours")
    #do whatever
else:
    _mygzs = [round(i,2) for i in np.arange(0.75,2.251,0.01)]
    _simpz_g = np.arange(1.,2.01,0.05)
    _model = _colour_shelf()
    