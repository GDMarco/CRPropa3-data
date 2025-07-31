import numpy as np
from scipy.integrate import cumulative_trapezoid, romb, quad, trapezoid
import os
from units import eV, Mpc, c_light, c_squared, ccm
import gitHelp as gh
import matplotlib.pyplot as plt

def calc_rate_eps(eps, xs, gamma, field, z=0, cdf=False):
    """
    Calculate the interaction rate for given tabulated cross sections against an isotropic photon background.
    The tabulated cross sections need to be of length n = 2^i + 1 and the tabulation points log-linearly spaced.

    eps   : tabulated photon energies [J] in nucleus rest frame
    xs    : tabulated cross sections [m^2]
    gamma : (array of) nucleus Lorentz factors
    field : photon background, see photonField.py
    z     : redshift
    cdf   : calculate cumulative differential rate

    Returns :
        interaction rate 1/lambda(gamma) [1/Mpc] or
        cumulative differential rate d(1/lambda)/d(s_kin) [1/Mpc/J^2]
    """
    F = cumulative_trapezoid(x=eps, y=eps * xs, initial=0)
    n = field.getDensity(np.outer(1. / (2 * gamma), eps), z)
    if cdf:
        y = n * F / eps**2
        return cumulative_trapezoid(x=eps, y=y, initial=0) / np.expand_dims(gamma, -1) * Mpc
    else:
        y = n * F / eps
        dx = mean_log_spacing(eps)
        return romb(y, dx=dx) / gamma * Mpc

def calc_rate_s(s_kin, xs, E, field, z=0, cdf=False):
    """
    Calculate the interaction rate for given tabulated cross sections against an isotropic photon background.
    The tabulated cross sections need to be of length n = 2^i + 1 and the tabulation points log-linearly spaced.

    s_kin : tabulated (s - m**2) for cross sections [J^2]
    xs    : tabulated cross sections [m^2]
    E     : (array of) cosmic ray energies [J]
    field : photon background, see photonField.py
    z     : redshift
    cdf   : calculate cumulative differential rate

    Returns :
        interaction rate 1/lambda(gamma) [1/Mpc] or
        cumulative differential rate d(1/lambda)/d(s_kin) [1/Mpc/J^2]
    """
    if cdf:
        # precalculate the field integral if it not exists and load it afterwards
        calculateDensityIntegral(field)
        file = "temp/fieldDensity/" + field.name + ".txt"
        densityIntegral = np.loadtxt(file)

        print("max/min density integral: ", np.max(densityIntegral), np.min(densityIntegral))

        # interpolate
        I = np.zeros((len(E), len(s_kin)))
        print("Massless Ishape: ", I.shape)
        
        for j in range(len(E)):
            I[j,:] = np.interp(s_kin / 4 / E[j], densityIntegral[:,0], densityIntegral[:,1])

        print("I max/min (SI): ", np.max(I), np.min(I))

        # calculate cdf
        y = np.array([xs * s_kin for i in range(len(E))]) * I
        print("y shape: ", y.shape)
        print("y max/min (SI): ", np.max(y), np.min(y))
        cdf = cumulative_trapezoid(y = y, x = s_kin, initial=0) / 8 / np.expand_dims(E, -1)**2 * Mpc    
        
        print("cdf shape: ", cdf.shape)
        print("cdf max/min (SI): ", np.max(cdf), np.min(cdf))
        return cdf
    
    else:
        # beta
        F = cumulative_trapezoid(x=s_kin, y=s_kin * xs, initial=0)
        n = field.getDensity(np.outer(1. / (4 * E), s_kin), z)
        
        y = n * F / s_kin
        
        ds = mean_log_spacing(s_kin)
        I = romb(y, dx=ds) / 2 / E * Mpc
        
        return I

def getMomenta(s_grid, E_grid, field):
    """
    Parameters
    ----------
    s_grid : tabulated (s - m**2) for cross sections [J^2], it should be a 2D array coming from np.meshgrid(s_kin, E) 
    E_grid : grid of cosmic ray energies [J]
    field : massive  neutrino background, see neutrinoField.py

    Returns
    -------
    momenta matrix for all the combinations of s and E

    """
    mass = field.mass
    p = s_grid / 4 / E_grid / c_light - mass * mass * c_squared * c_light * E_grid / s_grid
    
    # chec for the minimum momentum
    
    if np.any(p < 0):
        #print("Warning: some p values are negative. Fixing them to zero.")        
        p = np.clip(p, 0, None)
        
    return p

def betaRelative(s_kin, mass, field): 
    '''
    Parameters
    ----------
    s_grid : grid of center-of-mass energy squared [eV**2]?
    mass : mass of the propagating neutrino (in kg!)
    field : neutrino background (to get its mass)

    Returns
    -------
    relative Beta between the propagating (massive) neutrino and (massive) neutrino background 
    '''
    
    ratio = field.mass * mass * c_squared * c_squared / s_kin 
    #s_kin = (s - (field.mass * field.mass + mass * mass) * c_squared * c_squared) !
    return np.sqrt(1 - 4 * ratio * ratio)

def calc_rate_s_momenta(s_kin, xs, E, mass, field, z=0, cdf=False):
    """
    Calculate the interaction rate for given tabulated cross sections against an isotropic massive neutrino background.
    The tabulated cross sections need to be of length n = 2^i + 1 and the tabulation points log-linearly spaced.

    s_kin : tabulated (s - m**2) for cross sections [J^2]
    xs    : tabulated cross sections [m^2]
    E     : (array of) cosmic ray energies [J]
    field : massive neutrino background, see neutrinoField.py
    z     : redshift
    cdf   : calculate cumulative differential rate

    Returns :
        interaction rate 1/lambda(gamma) [1/Mpc] or
        cumulative differential rate d(1/lambda)/d(s_kin) [1/Mpc/J^2]
    """

    if cdf:
        
        # precalculate the field integral if it not exists and load it afterwards
        calculateDensityMomentumIntegral(field, s_kin, E)
        file = "temp/fieldDensity/" + field.name + ".txt"
        densityIntegral = np.loadtxt(file)
        
        print("max/min density integral: ", np.max(densityIntegral), np.min(densityIntegral))
        # interpolate
        I = np.zeros((len(E), len(s_kin)))
        
        print("Ishape: ", I.shape)
        
        for j in range(len(E)):
            p = getMomenta(s_kin, E[j], field)
            #print("p shape: ", p.shape)
            I[j,:] = np.interp(p, densityIntegral[:,0], densityIntegral[:,1])
        
        print("I max/min (SI): ", np.max(I), np.min(I))
        
        betaRel = betaRelative(s_kin, mass, field)

        # calculate cdf
        y = np.array([xs * s_kin * betaRel for i in range(len(E))]) * I
        print("y shape: ", y.shape)
        print("y max/min (SI): ", np.max(y), np.min(y))
        cdf = cumulative_trapezoid(y = y, x = s_kin, initial=0) / 8 / np.expand_dims(E, -1)**2 * Mpc    
        
        print("cdf shape: ", cdf.shape)
        print("cdf max/min (SI): ", np.max(cdf), np.min(cdf))
        return cdf
    
    else:

        betaRel = betaRelative(s_kin, mass, field)
        print("maxbetaRel: ", max(betaRel), ", min:", min(betaRel))
        F = cumulative_trapezoid(x=s_kin, y=betaRel * s_kin * xs, initial=0)

        s_grid, E_grid = np.meshgrid(s_kin, E)
        
        p_grid = getMomenta(s_grid, E_grid, field)
        
        #nonzero_p = p_grid[p_grid > 0]
        #min_p_gridNo0 = np.min(nonzero_p) if nonzero_p.size > 0 else None
        
        n = field.getDensity(p_grid, z)
        
        denom = np.sqrt(c_light * p_grid * np.sqrt((p_grid * c_light)**2 + (field.mass * c_squared)**2))
        
        #nonzero_denom = denom[denom > 0]
        #min_denomNo0 = np.min(nonzero_denom) if nonzero_denom.size > 0 else None
        
        y = np.where(denom != 0, (n * F) / denom, 0.0)
        
        nonzero_y = y[y > 0]
        min_nonzero = np.min(nonzero_y) if nonzero_y.size > 0 else None
        
        ds = mean_log_spacing(s_kin)
        I = romb(y, dx=ds, axis=1) / 8 / E / E * Mpc
        
        return I

'''
def calc_doubleIntegration_Is(s_kin, xs, mass, field):
    
    betaRel = betaRelative(s_kin, mass, field)
    F = trapezoid(x=s_kin, y=betaRel * s_kin * xs)
    
    return F
    
def calc_doubleIntegration_Ip():
'''    
    
def calculateDensityIntegral(field):
    """ 
        Precalculate the integral over the density 
        int_{Emin}^{Emax} n(eps) / eps^2  deps 
        and save as a file.

        field : photon background, see photonField.py
    """

    # check if file already exist
    folder = "temp/fieldDensity/"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    file = folder + field.name + ".txt"
    if os.path.isfile(file):
        return # file already existst no calculation necessary

    # precalc the photon density integral 
    Emax = field.getEmax()  # to change this Emin, according to the max skin set!!!!
    Emin =  1e4 / 4 / 1e27 * eV # min(s_kin) / 4 / max(E_e)
    alpha = np.logspace(np.log10(Emin), np.log10(Emax), 10000) # lower boundary of the integral.

    # calculate integral
    I_gamma = np.zeros_like(alpha)
    for i in range(len(alpha)):
        I_gamma[i] = quad(lambda E: field.getDensity(E) / E**2, a = alpha[i], b = Emax, full_output=1)[0]

    # save file
    header = "# Integrated spectral photon density.\n" 
    header += "# Integral n(e)/e^2 de from eMin to eMax, where eMax is the maximal photon energy of the background \n"
    try: 
        git_hash = gh.get_git_revision_hash()
        header += "# Produced with crpropa-data version: "+git_hash+"\n"
        header += "# eMin [eV]\tintegral\n"
    except:
        header += "# eMin [eV]\tintegral\n"
    data = np.c_[alpha, I_gamma]
    fmt = '%.4e\t%8.7e'
    np.savetxt(file, data, fmt = fmt, header = header)
    
def calculateDensityMomentumIntegral(field, s_kin, E, z=0):
    """ 
        Precalculate the integral over the density 
        int_{Pmin}^{Pmax} n(p) / eps^2  dp 
        and save as a file.

        field : massive neutrino background, see neutrinoField.py
    """

    # check if file already exist
    folder = "temp/fieldDensity/"
    if not os.path.isdir(folder):
        os.makedirs(folder)
    file = folder + field.name + ".txt"
    
    # for now deactivated!!
    #if os.path.isfile(file):
    #    return # file already existst no calculation necessary

    # precalc the photon density integral 
    Pmax = field.getPmax(z)  
    Pmin = field.getPmin(z) # getMomenta(s_kin[0], E[-1], field) # min(field.getPmin(z), )
    
    print("pmin for calculate density mom integral: ", Pmin / eV * c_light)
    
    alpha = np.logspace(np.log10(Pmin), np.log10(Pmax), 10000) # lower boundary of the integral.

    # calculate integral
    I_gamma = np.zeros_like(alpha)
    for i in range(len(alpha)):
        I_gamma[i] = quad(lambda p: field.getDensity(p) / p / c_light / np.sqrt(((p * c_light) ** 2 + (field.mass * c_squared) ** 2)), a = alpha[i], b = Pmax, full_output=1)[0]

    # save file
    header = "# Integrated momentum neutrino density.\n" 
    header += "# Integral n(p)/eps^2 de from pMin to pMax, where pMax is the maximal neutrino momentum of the background (at a certain redshift) \n"
    try: 
        git_hash = gh.get_git_revision_hash()
        header += "# Produced with crpropa-data version: "+git_hash+"\n"
        header += "# pMin [eV / c]\tintegral\n"
    except:
        header += "# pMin [eV / c]\tintegral\n"
    data = np.c_[alpha, I_gamma]
    fmt = '%.4e\t%8.7e'
    np.savetxt(file, data, fmt = fmt, header = header)

def mean_log_spacing(x):
    """ <Delta log(x)> """
    return np.mean(np.diff(np.log(x)))


def romb_truncate(x, n):
    """ Truncate array to largest size n = 2^i + 1 """
    i = int(np.floor(np.log2(n))) + 1
    return x[0:2**i + 1]

def romb_pad_zero(x, n):
    """ Pad array with zeros """
    npad = n - len(x)
    return np.r_[x, np.zeros(npad)]

def romb_pad_logspaced(x, n):
    """ Pad array with log-linear increasing values """
    npad = n - len(x)
    dlx = np.mean(np.diff(np.log(x)))
    xpad = x[-1] * np.exp(dlx * np.arange(1, npad + 1))
    return np.r_[x, xpad]
