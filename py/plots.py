import os
import dataload as dld
import bmc_common as bmccom
import hzq,mlt,etg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.backends.backend_pdf import PdfPages

def out_folder(fnm):
    target = dld.startpath+"/data/"+fnm+"_output/"
    if not os.path.exists(target):
        os.makedirs(target)
    return target

def rows_of_interest(d):
    if len(d) == 1:
        print("There is only one object in the input. Proceeding...")
        ui = "1"
    else:
        ui = input(f"Selections must be in the range 1-{len(d)}: ")
    #look for the hyphen in the input - if there isn't one ui = -1 and check for valid integer
    ix = ui.find("-")
    iy = ui.find(",")#comma in the input for non-contiguous selections
    if ix < 0 and iy < 0: #return a single row
        try:
            d = d.iloc[int(ui) - 1,:]
            return d
        except (KeyError, TypeError, ValueError, IndexError):
            print("Sorry, there is an issue with your row selection. Please try again")
            return rows_of_interest(d)
    elif ix < 0 and iy >= 0:
        try:
            sel = ui.split(",")
            d = d.iloc[[int(seli) - 1 for seli in sel],:]
            return d
        except (KeyError, TypeError, ValueError, IndexError):
            print("Sorry, there is an issue with your row selection. Please try again")
            return rows_of_interest(d)
    else: #contiguous selection of rows
        try:
            a,b = int(ui[:ix]) - 1, int(ui[ix+1:])
            if a >= 0 and b > 0 and b >= a and b <= len(d): 
                return d.iloc[a:b ,:]
            else:
                print("Sorry, there is an issue with your row selection. Please try again")
                return rows_of_interest(d)
        except (KeyError, TypeError, ValueError, IndexError):
            print("Sorry, there is an issue with your row selection. Please try again")
            return rows_of_interest(d)

def make_plottable_sed(dic):
    w = dic['wavelengths']
    f = [i * 1.E6 * dic['scaling'] for i in dic['template']]
    sed = sorted(zip(w,f), key=lambda x: x[0])
    return sed

def plottext(d,phot,q,g,s,JF):
    l = len([i for i in q['e'] if type(i)!=str])
    limits = 0 if l == len(q['e']) else 1
    dof = l - 1.
    probs = 1 if 'Pq' in d else 0
    #coordinates
    if 'ra_d' in d:
        ra,dec = round(d['ra_d'],4),round(d['dec_d'],4)
    else:
        ra,dec = d['ra_s'],d['dec_s']
    txtpos = f"coordinates: {ra} {dec}\n"
    #Jfilter
    if "VIK" in JF:
        JF = "MKO"
    txtmag = "$J_{AB}$" + f"= {dld._flux2mag(phot[f'J_{JF}1']['f'],3631.)}\n"
    txts = f"MLT: {s['spt']}; " + "$\chi^2_{red}$=" + f"{s['chisq']/dof:.2g}\n"
    txtg = f"ETG: $z_f$ = {g['zf'][2:]}; $z$ = {g['z']:.2f}; " + "$\chi^2_{red}$=" + f"{g['chisq']/dof:.2g}\n"
    txtq = f"HZQ: model = {q['qm']}; $z$ = {q['qz']:.2f}; " + "$\chi^2_{red}$=" + f"{q['chisq']/dof:.2g}\n"
    #limit/prob combo   
    if limits and probs:
        txtfinal = (f"Wq: {d['Wq']:.3g};  Ws: {d['Ws']:.3g};  Wg: {d['Wg']:.3g} \n"
                    f"Pq: {d['Pq']:.2g}\n"
                    "Limits are not included in the $\chi^2$ calculation\n"
                    "but are included in the $P_q$ calculation.")
    elif limits and not probs:
       txtfinal = "Limits are not included in the $\chi^2$ calculation"
    elif not limits and probs:
        txtfinal = (f"Wq: {d['Wq']:.2g};  Ws: {d['Ws']:.2g};  Wg: {d['Wg']:.2g}\n"
                    f"Pq: {d['Pq']:.2g}\n"
                    "Limits are not included in the $\chi^2$ calculation\n"
                    "but are included in the $P_q$ calculation.")
    else:
        txtfinal = "All filters included in the $\chi^2$ calculation"  
        
    txt = (f"{txtpos}"
           f"{txtmag}"
           f"{txts}"
           f"{txtg}"
           f"{txtq}"
           f"{txtfinal}")
    return txt

def plot_params(ax): #get tick locators etc from axis size
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xmax = np.ceil(xmax)
    xmin = np.floor(xmin)
    xmajloc = xmax/10.
    xminloc = xmajloc/5
    ymajloc = 10.*np.floor(np.log10(ymax))
    yminloc = ymajloc/5.
    return (xmin,xmax,xmajloc,xminloc,ymajloc,yminloc)
        
def best_fit_sed_plot(d,phot,JF):
    bfq = hzq.best_fit_SED(phot,JF); qw,qf = zip(*make_plottable_sed(bfq))
    bfg = etg.best_fit_SED(phot,JF); gw,gf = zip(*make_plottable_sed(bfg))
    bfs = mlt.best_fit_SED(phot,JF); sw,sf = zip(*make_plottable_sed(bfs))
    #make plot
    fig = plt.figure(figsize=(8,5),dpi=100)
    ax1 = fig.add_subplot(111)
    #xmin = round(min(bfq['wavelengths'])-0.2,2) ; xmax = round(max(bfq['wavelengths'])+0.25,2)
    #ymin = round((1.E6*min(bfq['f']))-0.2,1); ymax = round((1.E6*max(bfq['f']))+0.5,1)
    #ax1.set_xlim(xmin,xmax); 
    ax1.set_xlabel(r'wavelength ($\mu$m)',fontsize=11)
    ax1.set_ylabel(r'flux ($\mu$Jy)',fontsize=11)
    ax1.plot(qw,qf,'k-',lw=2,label='quasar')
    ax1.plot(gw,gf,'r--',lw=2,label='galaxy')
    ax1.plot(sw,sf,'y:',lw=2,label='MLT')#plot templates
    #plot the source
    sourcew,sourcef,sourcee=[],[],[]
    for i,e in enumerate(bfq['e']):
        if type(e) != str:
            sourcew.append(bfq['wavelengths'][i])
            sourcef.append(bfq['f'][i]*1.E6)
            sourcee.append(e*1.E6)
    ax1.errorbar(sourcew,sourcef,sourcee, linestyle="None",color='b',capsize=4,marker='o',ms=4,label='photometry')
    #check the photometry dictionary for any limits bands - these won't have been included in the chisq fitting
    xmin,xmax,xmajloc,xminloc,ymajloc,yminloc = plot_params(ax1)    
    for b in phot:
        b_nodigits = bmccom.remove_digits(b); w = dld._eff_wavelength[b_nodigits]
        ax1.text(w,-ymajloc*0.5,f"{b_nodigits}",fontsize=7)
        if type(phot[b]['e']) == str:
            b_nodigits = bmccom.remove_digits(b); w = dld._eff_wavelength[b_nodigits]
            ax1.errorbar(w,phot[b]['f'],xerr=0.75*xminloc,yerr=yminloc, linestyle="None",color='b',uplims=1)
    ax1.tick_params(axis='both', which='major', labelsize=10)        
    ax1.xaxis.set_major_locator(tck.MultipleLocator(xmajloc))
    ax1.xaxis.set_minor_locator(tck.MultipleLocator(xminloc))
    ax1.yaxis.set_major_locator(tck.MultipleLocator(ymajloc))
    ax1.yaxis.set_minor_locator(tck.MultipleLocator(yminloc))
    ax1.set_ylim(bottom=-ymajloc)
    ax1.set_xlim(right=xmax)
    ax1.legend(frameon=0,fontsize=9)
    plottxt = plottext(d,phot,bfq,bfg,bfs,JF)
    ax1.text(xmax+xminloc,ymajloc,plottxt,fontsize=6)
    plot_params(ax1)
    plt.subplots_adjust(left=0.1,right=0.75)
    return fig


def posterior_plot(phot,dsinb,JF):
    Q = hzq.dW_dz(phot,JF)
    S = mlt.dW_dt(phot,dsinb,JF)
    G = etg.dW_dz(phot,JF)
    fig, ax = plt.subplots(3, 1 ,facecolor='white',figsize=(5,8),dpi=100)
    fig.tight_layout(pad=3.)
    #Q
    ax[0].plot(Q['z'],Q['vals'],'k-')
    ax[0].set_xlabel(r'$z$',fontsize=11)
    ax[0].set_ylabel(r'$\frac{d}{dz}W_q$',fontsize=11)
    ax[0].xaxis.set_major_locator(tck.MultipleLocator(0.25))
    ax[0].xaxis.set_minor_locator(tck.MultipleLocator(0.05))
    #G
    ax[1].plot(G['z'],G['vals'],'r-')
    ax[1].set_xlabel(r'$z$',fontsize=11)
    ax[1].set_ylabel(r'$\frac{d}{dz}W_g$',fontsize=11)
    ax[1].xaxis.set_major_locator(tck.MultipleLocator(0.25))
    ax[1].xaxis.set_minor_locator(tck.MultipleLocator(0.05))
    #S
    X = S['i']; Y = [0] + S['vals']
    ax[2].step(X,Y,'y-',where='pre')
    ax[2].plot([t - 0.5 for t in X[1:]] ,Y[1:],c='y',ls='None',marker='o',ms=2)    
    #for i,val in enumerate(X):
    #    ax[2].axvline(X[i],ymin=0,ymax=Y[i+1],lw=0.5,c='y',ls=':')
    ax[2].set_xlabel(r'spectral type',fontsize=11)
    ax[2].set_ylabel(r'$W_{SpT}$',fontsize=11)
    ax[2].xaxis.set_major_locator(tck.MultipleLocator(5))
    ax[2].xaxis.set_minor_locator(tck.MultipleLocator(1))
    ax[2].set_xticklabels(['M0']+S['t'][::5])
    plt.tight_layout()
    #print(S)  
    return fig

def make_plots(d,opath):
    pdfname = f"row_{d.name+1}.pdf"
    phot = dld._flux_dict(d)
    p1 = best_fit_sed_plot(d,phot,JF)
    p2 = posterior_plot(phot,d['sinb'],JF)
    out_pdf = PdfPages(opath+pdfname)
    for p in [p1,p2]:
        out_pdf.savefig(p)
    out_pdf.close()
    plt.close(p1)
    plt.close(p2)  
    return


if __name__ == "__main__":
    filename = dld.filename
    opath = out_folder(filename)
    JF = dld._Jfilt
    d = dld._sinb(dld.csv_load(filename))
    if len(d) > 1:
        print("Please enter rows of data file to be analysed.")
        print("You can enter a single number, comma-separated values, or a range of rows as first-last")
    d = rows_of_interest(d)
    #if there's just one row returned it's a series, if not a dataframe
    if isinstance(d, pd.DataFrame):
        d.apply(lambda x: make_plots(x,opath), axis=1)
    else:
        make_plots(d,opath)
    print(f"Done. Please check output in the folder {opath}")
    
