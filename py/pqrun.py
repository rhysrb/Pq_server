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

if __name__ == "__main__":
    filename = dld.filename
    d = dld.csv_load(filename)
    JF = dld._Jfilt
    dsinb = dld._sinb(d)
    print(f"Starting Pq calculation for {len(dsinb)} objects...") 
    dsinb = dsinb.progress_apply(lambda x: Pq(x), axis=1)
    message = dld.csv_save(dsinb)
    print(f"Pq calculation done! {message}")