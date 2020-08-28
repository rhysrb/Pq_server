import dataload as dld
#import bmc_common as bmc
import hzq,mlt,etg
import pandas as pd
from tqdm import tqdm; tqdm.pandas()

#Calculate the quasar probability from individual weights
def Pq(d):
    #sb = dld._sinb(d)
    phot = dld._flux_dict(d)
    Wq = hzq.W(phot,JF)
    Ws = mlt.W(phot,d['sinb'],JF)
    Wg = etg.W(phot,JF)
    Pq = Wq/(Wq+Ws+Wg)
    d['Ws'] = Ws
    d['Wg'] = Wg
    d['Wq'] = Wq
    d['Pq'] = Pq
    return d

def weights_and_prob(d,JF): #used in plots.py
    check = pd.Series(['Ws', 'Wg', 'Wq', 'Pq'])
    if sum(check.isin(d.index))==4 and not d[check].hasnans:
        return d['Ws'],d['Wg'],d['Wq'],d['Pq']
    else:
        phot = dld._flux_dict(d)
        Wq = hzq.W(phot,JF)
        Ws = mlt.W(phot,d['sinb'],JF)
        Wg = etg.W(phot,JF)
        Pq = Wq/(Wq+Ws+Wg)
        return Ws,Wg,Wq,Pq

if __name__ == "__main__":
    filename = dld.filename
    d = dld.csv_load(filename)
    JF = dld._Jfilt
    dsinb = dld._sinb(d)
    print(f"Starting Pq calculation for {len(dsinb)} objects...") 
    dsinb = dsinb.progress_apply(lambda x: Pq(x), axis=1)
    message = dld.csv_save(dsinb)
    print(f"Pq calculation done! {message}")
    
else: #for pipeline use - initialise variables here
    dld._Jfilt = "MKO_LAS"
    JF = dld._Jfilt
    _eff_wavelength = {'g_PS':0.481,'r_PS':0.616,\
            'i_PS': 0.750,'z_PS': 0.867,'y_PS': 0.961,\
            #note LSST & PS are very similar.
            'u_LSST':0.375,'g_LSST':0.474,'r_LSST':0.617,\
            'i_LSST':0.750,'z_LSST': 0.868,'y_LSST': 0.971,\
            'u_SDSS':0.359,'g_SDSS':0.464,'r_SDSS':0.612,\
            'i_SDSS': 0.744,'z_SDSS': 0.890,\
            'g_DEC':0.477,'r_DEC':0.637,'i_DEC':0.777,\
            'z_DEC': 0.916,'Y_DEC': 0.988,\
            'r_COS': 0.623,'i_COS':0.763,
            'z_COS': 0.909,'y_COS': 0.975,\
            'Z_MKO': 0.877,'Y_MKO': 1.020,'J_MKO': 1.252,'H_MKO': 1.645,'K_MKO':2.201,'KS_MKO':2.147,\
            'W1_WISE':3.353,'W2_WISE':4.603,\
            'O_EUC': 0.687,'Y_EUC': 1.073,'J_EUC': 1.343,'H_EUC': 1.734}