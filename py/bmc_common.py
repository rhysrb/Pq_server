#define a few things that are common to all three populations in the BMC calcuation.
#This includes the simpsons rule function, and likelihood functions.
import numpy as np

if __name__ == "__main__":
    print("Value of __name__ is:", __name__)
    print("Running bmc_common.py module")
    #do other test stuff...

#single quad combined with simpsons approximation over z is much faster
#than doublequad for ellipticals and quasars.
#x/y should automatically be even in length since we choose dz = 0.05
def simpson(x,y,l=False,u=False):
    l = x[0] if not l else l
    u = x[-1] if not u else u
    total = 0
    for i in range(0, len(x) - 2,2):
        if x[i] >= l and x[i+2] <= u:
            total += (y[i]+(4*y[i+1])+y[i+2])
    return (x[2]-x[0])*total/6.

#everything should be in flux units already
def liketerm(m,f,e):
    #deal with upper limits - this will be the case if e is a string
    if type(e) == str and e.endswith('sig'):
        nsig = float(e.replace('sig',''))
        sig = f/nsig
        return _liketerm_erf(m,f,sig)
    else:
        return _liketerm_gauss(m,f,e)

def _liketerm_gauss(m,f,e):
    exponent = (f-m)/e
    return np.exp(-0.5*(exponent**2.)) #normalisations cancel out

def _liketerm_erf(m,f,e):
    a = (f-m)/(e*np.sqrt(2.))
    return 0.5*(1.+my_erf(a))

#very accurate numerical approximation of error function from Abramowitz and Stegun
#(maximum error: 1.5×10−7)
def my_erf(x):
    # save the sign of x
    sign = np.sign(x)
    x = abs(x)
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)  


########some functions associated with minimum chi squared plotting####################
   
def remove_digits(b):
    usi = b.find("_") #remove digits from b after the first underscore (preserve the digits in W1W2)
    return b[:usi] + ''.join([i for i in b[usi:] if not i.isdigit()])

def seds_for_chisq(phot,Jfilter):
    bands,sed,errors = [],[],[]
    for b in phot: #compile sed of real flux values
        if phot[b]['f']:
            b_nodigits = remove_digits(b)
            bands.append(b_nodigits)
            sed.append(phot[b]['f'])
            errors.append(phot[b]['e'])
    return bands,sed,errors


#minimised chi-sq scale factor
def scalebest(tmp,flx,err):
    if len(tmp) != len(flx) or len(tmp) != len(err):
        print("Error! Lengths of inputs do not match.")
        raise SystemExit        
    num = 0.
    den = 0.
    for i in range(0,len(tmp)):
        if flx[i] > -9.E8 and type(err[i]) != str:
            num += (tmp[i] * flx[i] / (err[i]**2.))
            den += ((tmp[i]**2.)/(err[i]**2.))
    return num/den


#work out chi-sq value for each model
def chisq(tmp,flx,err,sf):
    return ((flx - sf*tmp)/err)**2.

