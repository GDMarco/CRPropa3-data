from __future__ import division
import numpy as np
import interactionRate
import os
import gitHelp as gh
import math
from crpropa import eV, mass_electron, c_light, h_planck, GeV, c_squared, cm
from calc_all import reduced_fields, fields_CnuB, fields_massiveCnuB, cmb
from units import me2, sigmaThomson, alpha, mm2, mt2, mW2, mZ2, mUp2, mDown2
from units import mCharm2, mStrange2, mBottom2, mTop2, mNu2, Gf, gammaZ2, sW2

def F1exp0(M, s, sthr):
    y = s / M
    ythr = sthr / M

    if (y < ythr): 
        # expansion for s << M of F1 
        F1 = 1./3. - y/6. + y*y/10. - y*y*y/15. + y*y*y*y/21. - y*y*y*y*y/28.
        
    else: 
        
        F1 = (y*y+2.*y-2.*(1.+y)*np.log1p(y))/y**3
        
    return F1

def F2exp0(M, s, sthr):
    y = s / M
    ythr = sthr / M

    if (y < ythr): 
        # expansion for s << M of F1 
        F2 = -2./3. + y/6. - y*y/15. + y*y*y/30. - 2./105.*y*y*y*y + y*y*y*y*y/84.
        
    else:
        
        F2 = (3.*y*y+2.*y-2.*(1.+y)*(1.+y)*np.log1p(y))/y**3
    
    return F2 

def sigmaNuElGamma(s): #,massLep,alpha,Gf,massW
    """Neutrino-Photon interaction cross section, see D. Seckel (1997). Electron neutrinos."""
    smin=(np.sqrt(mW2)+np.sqrt(me2))**2
    if (s < smin):
        return 0.
    logFactor = mW2*(s/mW2-1.)**2./me2/(s/mW2)
    return np.sqrt(2.)*alpha*Gf*(2.*(1.-1./(s/mW2))*(1.+2./(s/mW2)**2.-1./(s/mW2)**2*np.log((s/mW2)))+1./(s/mW2)*(1.-2./(s/mW2)+2./(s/mW2)**2)*np.log(logFactor))

def sigmaNuMuGamma(s): #,massLep,alpha,Gf,massW
    """Neutrino-Photon interaction cross section, see D. Seckel (1997). Muon neutrinos."""
    smin=(np.sqrt(mW2)+np.sqrt(mm2))**2
    if (s < smin):
        return 0.
    logFactor = mW2*(s/mW2-1.)**2./mm2/(s/mW2)
    return np.sqrt(2.)*alpha*Gf*(2.*(1.-1./(s/mW2))*(1.+2./(s/mW2)**2.-1./(s/mW2)**2*np.log((s/mW2)))+1./(s/mW2)*(1.-2./(s/mW2)+2./(s/mW2)**2)*np.log(logFactor))

def sigmaNuTauGamma(s): #,massLep,alpha,Gf,massW
    """Neutrino-Photon interaction cross section, see D. Seckel (1997). Tau neutrinos."""
    smin=(np.sqrt(mW2)+np.sqrt(mt2))**2
    if (s < smin):
        return 0.
    logFactor = mW2*(s/mW2-1.)**2./mt2/(s/mW2)
    return np.sqrt(2.)*alpha*Gf*(2.*(1.-1./(s/mW2))*(1.+2./(s/mW2)**2.-1./(s/mW2)**2*np.log((s/mW2)))+1./(s/mW2)*(1.-2./(s/mW2)+2./(s/mW2)**2)*np.log(logFactor))

################### 

def funcNuNuxZres(Qf, t3, nf, s, mf2):
    ''' mf2 = (mass_f[kg] * c_light**2)**2 '''
    smin = 4.*mf2
    if (s < smin):
        return 0.
    
    Pz = mZ2*mZ2 / ((s-mZ2)**2+gammaZ2*mZ2)
    par = (t3*t3-2.*t3*Qf*sW2+2.*Qf*Qf*sW2*sW2)
    return 2.*Gf*Gf/3./math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*nf*Pz*s*par

def sigmaNuNuxZresEl(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    
    #leptons
    Qf = 1.
    t3 = 0.
    nf = 1.
    mf2 = me2
    NuNuxEl = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxEl
    
def sigmaNuNuxZresMu(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    
    Qf = 1.
    t3 = 0.
    nf = 1.
    mf2 = mm2
    NuNuxMu = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxMu 
    
def sigmaNuNuxZresTa(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    
    Qf = 1.
    t3 = 0.
    nf = 1.
    mf2 = mt2
    NuNuxTa = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxTa

def sigmaNuNuxZresUp(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    #quarks
    Qf = 2./3.
    t3 = 1./2.
    nf = 3.
    mf2 = mUp2
    NuNuxUp = funcNuNuxZres(Qf, t3, nf, s, mf2)    
    return NuNuxUp
    
def sigmaNuNuxZresDown(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    
    Qf = -1./3.
    t3 = -1./2.
    nf = 3.
    mf2 = mDown2
    NuNuxDown = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxDown

def sigmaNuNuxZresCharm(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    Qf = 2./3.
    t3 = 0.
    nf = 3.
    mf2 = mCharm2
    NuNuxCharm = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxCharm

def sigmaNuNuxZresStrange(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    Qf = -1./3.
    t3 = 0.
    nf = 3.
    mf2 = mStrange2
    NuNuxStrange = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxStrange 

def sigmaNuNuxZresTop(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    Qf = 2./3.
    t3 = 0.
    nf = 3.
    mf2 = mTop2
    NuNuxTop = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxTop

def sigmaNuNuxZresBottom(s): 
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    Qf = -1./3.
    t3 = 0.
    nf = 3.
    mf2 = mBottom2
    NuNuxBottom = funcNuNuxZres(Qf, t3, nf, s, mf2)
    return NuNuxBottom

def sigmaNuiNuxjZel(s):
    """Neutrino_i-(anti)neutrino_j Z-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.2"""
    #smin = 4.*mNu2
    #if (s < smin):
    #    return 0.
    
    sthr = 5e15 * eV * eV
    F1val = F1exp0(mZ2, s, sthr)
    return 0.5*Gf*Gf/math.pi/h_planck/h_planck/c_light*(2.*math.pi)*(2.*math.pi)/c_light*s*F1val 
    
def sigmaNuNuxZel(s):
    """Neutrino-(anti)neutrino interaction Z-exchange interaction cross section, see Roulet (1992) Eq. 2.3"""
    
    #smin = mNu2
    #if (s < smin):
    #    return 0.
    
    Pz = mZ2*mZ2 / ((s-mZ2)**2+gammaZ2*mZ2) 
    
    sthr = 5e15 * eV * eV
    F2val = F2exp0(mZ2, s, sthr)
    CS = 0.5*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*Pz*(s-mZ2)/mZ2*s*F2val
    
    return CS

def sigmaNuiNuxjWElMux(s):
    """Neutrino_i-(anti)neutrino_j interaction W-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.4"""
    
    smin = (np.sqrt(me2) + np.sqrt(mm2))**2.
    if (s < smin):
        return 0.
    
    sthr = 1e15 * eV * eV
    F1val = F1exp0(mW2, s, sthr)
    return 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*F1val

def sigmaNuiNuxjWElTax(s):
    """Neutrino_i-(anti)neutrino_j interaction W-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.4"""
    
    smin = (np.sqrt(me2) + np.sqrt(mt2))**2.
    if (s < smin):
        return 0.
   
    sthr = 5e17 * eV * eV
    F1val = F1exp0(mW2, s, sthr)
    return 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*F1val

def sigmaNuiNuxjWMuElx(s):
    """Neutrino_i-(anti)neutrino_j interaction W-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.4"""
    
    smin = (np.sqrt(me2) + np.sqrt(mm2))**2.
    if (s < smin):
        return 0.
   
    sthr = 5e15 * eV * eV
    F1val = F1exp0(mW2, s, sthr)
    return 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*F1val

def sigmaNuiNuxjWMuTax(s):
    """Neutrino_i-(anti)neutrino_j interaction W-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.4"""
    
    smin = (np.sqrt(mt2) + np.sqrt(mm2))**2.
    if (s < smin):
        return 0.
    
    sthr = 5e15 * eV * eV
    F1val = F1exp0(mW2, s, sthr)
    return 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*F1val

def sigmaNuiNuxjWTaElx(s):
    """Neutrino_i-(anti)neutrino_j interaction W-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.4"""
    
    smin = (np.sqrt(me2) + np.sqrt(mt2))**2.
    if (s < smin):
        return 0.
    
    sthr = 5e15 * eV * eV
    F1val = F1exp0(mW2, s, sthr)
    return 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*F1val

def sigmaNuiNuxjWTaMux(s):
    """Neutrino_i-(anti)neutrino_j interaction W-exchange t-channel interaction cross section, see Roulet (1992) Eq. 2.4"""
    
    smin = (np.sqrt(mt2) + np.sqrt(mm2))**2.
    if (s < smin):
        return 0.

    sthr = 5e15 * eV * eV
    F1val = F1exp0(mW2, s, sthr)
    return 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*F1val

def sigmaNuNuxWZEl(s): 
    """Neutrino-(anti)neutrino interaction cross section, see Roulet (1992) Eq. 2.5, INTERFERENCE OF Eq. 2.1 and Eq. 2.5"""
    
    smin = mZ2 # to check it! 
    if (s < smin):
        return 0.
    
    sthr = 5e15 * eV * eV
    F2val = F2exp0(mW2, s, sthr)
    Pz = mZ2*mZ2 / ((s-mZ2)**2+gammaZ2*mZ2)
    CS = 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*(sW2-0.5)*Pz*(s-mZ2)/mZ2*s*F2val
    return CS
    
def sigmaNuNuxWZMu(s): 
    """Neutrino-(anti)neutrino interaction cross section, see Roulet (1992) Eq. 2.5, INTERFERENCE OF Eq. 2.1 and Eq. 2.5"""
    
    smin = mZ2
    if (s < smin):
        return 0.

    sthr = 5e17 * eV * eV
    F2val = F2exp0(mW2, s, sthr)
    Pz = mZ2*mZ2 / ((s-mZ2)**2+gammaZ2*mZ2)
    CS = 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*(sW2-0.5)*Pz*(s-mZ2)/mZ2*s*F2val
    return CS
    
def sigmaNuNuxWZTa(s): 
    """Neutrino-(anti)neutrino interaction cross section, see Roulet (1992) Eq. 2.5, INTERFERENCE OF Eq. 2.1 and Eq. 2.5"""
    
    smin = mZ2
    if (s < smin):
        return 0.
   
    sthr = 5e15 * eV * eV
    F2val = F2exp0(mW2, s, sthr)
    Pz = mZ2*mZ2 / ((s-mZ2)**2+gammaZ2*mZ2)
    CS = 2.*Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*(sW2-0.5)*Pz*(s-mZ2)/mZ2*s*F2val
    return CS
    
def sigmaNuNuxZProd(s):
    """Neutrino-(anti)neutrino (t-channel lepton exchange + s-channel Z-exchange) interaction cross section, W bosons production, see Roulet (1992) new version, Eq. 2.7"""
    smin = 4.*mZ2
    if (s < smin):
        return 0.
    
    beta = np.sqrt(1.-4.*mZ2/s)
    L = np.log((1.+beta)/(1.-beta))
    y = s/mZ2
    
    return Gf*Gf/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*mZ2*beta/(y-2)*(2./y-1.+(4.+y*y)/2./y/y/beta*L)


def sigma_ZZ_incl_Rhorry(s_joule2):
    
    smin = 4.*mZ2
    if (s_joule2 < smin):
        return 0.
    
    # constants (redifined)
    GeV_to_J = 1.602176634e-10       # J per GeV
    pb_to_m2 = 1e-40                 # 1 pb = 1e-40 m2

    mz = 91.1876                     # Z mass in GeV
    ALPHA = 1/137.035999084          # fine-structure constant
    SW2 = 0.23126
    gLnu = 0.5 * np.sqrt(1.0 / (SW2 * (1.0 - SW2)))  # left-handed coupling
    hbarc2 = 0.389379e9              # (ħc)^2 in pb GeV2
    pi = np.pi

    # convert s from J2 to GeV2 
    shat = s_joule2 / (GeV_to_J ** 2)

    # XS computation 
    y = shat / mz**2
    sqrt_term = np.sqrt((-4 + y) * y)

    sigma = (
        512.0 / (-2 + y) / np.sqrt((-4 + y) * y)
        * (
            (
                (-np.log(-1 + (-2 + y) / np.sqrt((-4 + y) * y))
                 + np.log(1 + (-2 + y) / np.sqrt((-4 + y) * y)))
                * (4 + y**2) / 2.0
            )
            + 2 * np.sqrt((-4 + y) * y)
            - np.sqrt((-4 + y) * y**3)
        )
    )

    ps_factor = (1.0 / (16.0 * pi)) * np.sqrt(1.0 - 4.0 / y)
    prefactor = (ALPHA**2) * (pi**2) * (gLnu**2) * (np.conj(gLnu)**2)
    flux = 1.0 / (2.0 * shat)
    sym_fac = 2.0

    sigma_pb = sigma * (ps_factor * prefactor.real * flux / sym_fac) * hbarc2

    # convert pb to m2
    sigma_m2 = sigma_pb * pb_to_m2
    return sigma_m2.real

def sigmaNuNuxWProd(s):
    """Neutrino-(anti)neutrino (t-channel lepton exchange + s-channel Z-exchange) interaction cross section, Z bosons production, see Roulet (1992) Eq. 2.6"""
    smin = 4.*mW2
    if (s < smin):
        return 0.
    
    beta = np.sqrt(1.-4.*mW2/s)
    L = np.log((1.+beta)/(1.-beta))
    y = s/mW2
    
    return Gf*Gf/12./math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*beta*(beta*beta*mW2*mW2/(s-mZ2)**2*(12.+20.*y+y*y)+2.*mW2/(s-mW2)/y/y*(24.+28.*y-18.*y*y-y**3+48./beta/y*(1.+2.*y)*L)+(y*y+20.*y-48.-48./beta/y*L*(2.-y))/y/y)

def sigmaNuiNujZel(s):
    """(anti)Neutrino_i-(anti)neutrino_j Z-exchange t-channel elastic scatter cross section, see Roulet (1992) Eq. 2.7"""
    '''
    smin = 4.*mNu2
    if (s < smin):
        return 0.
    '''
    return Gf*Gf*mZ2*0.5/math.pi/h_planck/h_planck/c_light/c_light/(2.*math.pi)/(2.*math.pi)*s/(s+mZ2)

def sigmaNuNuel(s):
    """(anti)Neutrino-(anti)neutrino u-channel elastic scatter cross section, see Roulet (1992) Eq. 2.8"""
    '''
    smin = 4.*mNu2
    if (s < smin):
        return 0.
    '''
    return Gf*Gf*mZ2/2./math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*(s/(s+mZ2)+2.*mZ2/(2.*mZ2+s)*np.log(1.+s/mZ2))

def sigmaNuNuxZresNu(s):
    """Neutrino-(anti)neutrino Z-exchange s-channel interaction cross section, see Roulet (1992) Eq. 2.1"""
    '''
    smin = 4.*mNu2
    if (s < smin):
        return 0.
    '''
    # expression from Rhorry's code 
    # Rhorry's constant definitions
    
    J_to_GeV = 1.0 / 1.602176634e-10
    s = s * J_to_GeV**2 
    
    pb_to_cm2 = 1e-36
    
    mz = 91.15348062
    gz = 2.494266379
    gf = 1.16638e-5
    MZ2C = complex(mz * mz, -gz * mz)
    t3 = 0.5
    hbarc2 = 3.8937966e8
    prop = 1.0 / (s - MZ2C)
    spz = prop * prop.conjugate() * (mz ** 4) * s
        
    sigma = 2.0 * (gf ** 2) * spz.real * (t3 ** 2) / (3.0 * np.pi)
    return sigma * hbarc2 * pb_to_cm2 * cm**2

###################
sigmaNuNuxZres = [
    sigmaNuNuxZresEl,
    sigmaNuNuxZresMu, 
    sigmaNuNuxZresTa,
    sigmaNuNuxZresUp,
    sigmaNuNuxZresDown,
    sigmaNuNuxZresCharm,
    sigmaNuNuxZresStrange,
    sigmaNuNuxZresTop,
    sigmaNuNuxZresBottom
    ]

sigmaNuNuxWZ = [
    sigmaNuNuxWZEl,
    sigmaNuNuxWZMu,
    sigmaNuNuxWZTa
    ]

sigmaNuiNuxjW = [
    sigmaNuiNuxjWElMux,
    sigmaNuiNuxjWElTax,
    sigmaNuiNuxjWMuElx,
    sigmaNuiNuxjWMuTax,
    sigmaNuiNuxjWTaElx,
    sigmaNuiNuxjWTaMux
    ]

def getTabulatedXS(sigma, skin):
    """ Get cross section for tabulated s_kin """
    if sigma in (sigmaNuElGamma, sigmaNuMuGamma, sigmaNuTauGamma):
        # photon-neutrino interaction
        return np.array([sigma(s) for s in skin + mNu2])
    if sigma in (sigmaNuNuel, sigmaNuNuxWZEl,
                 sigmaNuNuxWZMu,
                 sigmaNuNuxWZTa, sigmaNuiNujZel, 
                 sigmaNuNuxZel, sigmaNuNuxZresEl,
                 sigmaNuNuxZresMu, 
                 sigmaNuNuxZresTa,
                 sigmaNuNuxZresUp,
                 sigmaNuNuxZresDown,
                 sigmaNuNuxZresCharm,
                 sigmaNuNuxZresStrange,
                 sigmaNuNuxZresTop,
                 sigmaNuNuxZresBottom, sigmaNuiNuxjWElMux,
                 sigmaNuiNuxjWElTax,
                 sigmaNuiNuxjWMuElx,
                 sigmaNuiNuxjWMuTax,
                 sigmaNuiNuxjWTaElx,
                 sigmaNuiNuxjWTaMux, sigmaNuiNuxjZel,
                 sigmaNuNuxWProd, sigmaNuNuxZProd):  
        # neutrino-neutrino interaction
        return np.array([sigma(s) for s in skin + 4.*mNu2 ])
    return False

def getTabulatedEffectiveXS(sigma, skin, mass, field):
    """ Get cross section for tabulated s_kin for massive propagating neutrinos interacting with massive neutrino background."""
    if sigma in (sigmaNuElGamma, sigmaNuMuGamma, sigmaNuTauGamma):
        # photon-neutrino interaction
        return np.array([sigma(s) for s in skin + mass * mass * c_squared * c_squared])
    if sigma in (sigmaNuNuel, sigmaNuNuxWZEl,
                 sigmaNuNuxWZMu,
                 sigmaNuNuxWZTa, sigmaNuiNujZel, 
                 sigmaNuNuxZel, sigmaNuNuxZresEl,
                 sigmaNuNuxZresMu, 
                 sigmaNuNuxZresTa,
                 sigmaNuNuxZresUp,
                 sigmaNuNuxZresDown,
                 sigma_ZZ_incl_Rhorry, # new computations
                 sigmaNuNuxZresNu, # new computations
                 sigmaNuNuxZresCharm,
                 sigmaNuNuxZresStrange,
                 sigmaNuNuxZresTop,
                 sigmaNuNuxZresBottom, sigmaNuiNuxjWElMux,
                 sigmaNuiNuxjWElTax,
                 sigmaNuiNuxjWMuElx,
                 sigmaNuiNuxjWMuTax,
                 sigmaNuiNuxjWTaElx,
                 sigmaNuiNuxjWTaMux, sigmaNuiNuxjZel,
                 sigmaNuNuxWProd, sigmaNuNuxZProd,
                 sigma_ZZ_incl_Rhorry, sigmaNuNuxZresNu):  
        # neutrino-neutrino interaction
        return np.array([sigma(s) for s in skin + (mass * mass + field.mass * field.mass) * c_squared * c_squared])
    return False

def getSmin(sigma):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction """

    return {
            
            sigmaNuElGamma: (np.sqrt(mW2)+np.sqrt(me2))**2,
            sigmaNuMuGamma: (np.sqrt(mW2)+np.sqrt(mm2))**2,
            sigmaNuTauGamma: (np.sqrt(mW2)+np.sqrt(mt2))**2,
            
            #'''
            sigmaNuNuxZel: 4.*mNu2,
            sigmaNuiNuxjZel: 4.*mNu2,
            sigmaNuNuel: 4.*mNu2, 
            sigmaNuiNujZel: 4.*mNu2,
        
            sigmaNuNuxWProd: 4.*mW2,
            sigmaNuNuxZProd: 4.*mZ2,
            sigma_ZZ_incl_Rhorry: 4.*mZ2,
            sigmaNuNuxWZEl: 4.*me2,
            sigmaNuNuxWZMu: 4.*mm2,
            sigmaNuNuxWZTa: 4.*mt2,
            
            sigmaNuNuxZresEl: 4.*me2,
            sigmaNuNuxZresMu: 4.*mm2,
            sigmaNuNuxZresTa: 4.*mt2,
            sigmaNuNuxZresUp: 4.*mUp2,
            sigmaNuNuxZresDown: 4.*mDown2,
            sigmaNuNuxZresCharm: 4.*mCharm2,
            sigmaNuNuxZresStrange: 4.*mStrange2,
            sigmaNuNuxZresTop: 4.*mTop2,
            sigmaNuNuxZresBottom: 4.*mBottom2,
            
            sigmaNuiNuxjWElMux: (np.sqrt(me2) + np.sqrt(mm2))**2.,
            sigmaNuiNuxjWElTax: (np.sqrt(me2) + np.sqrt(mt2))**2.,
            sigmaNuiNuxjWMuElx: (np.sqrt(me2) + np.sqrt(mm2))**2.,
            sigmaNuiNuxjWMuTax: (np.sqrt(mt2) + np.sqrt(mm2))**2.,
            sigmaNuiNuxjWTaElx: (np.sqrt(me2) + np.sqrt(mt2))**2.,
            sigmaNuiNuxjWTaMux: (np.sqrt(mt2) + np.sqrt(mm2))**2.
            #'''
            }[sigma]

def getEffectiveSmin(sigma, mass, field):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction, for interacting massive neutrinos """
    
    if (field.name == 'CMB' or field.name == 'IRB_Saldana21' or field.name == 'URB_Nitu21'):  
        fieldMass = 0 
    else: 
        
        fieldMass = field.mass
    
    return {
            sigmaNuNuxZel: (mass * mass + fieldMass * fieldMass) * c_squared * c_squared,
            sigmaNuiNuxjZel: (mass * mass + fieldMass * fieldMass) * c_squared * c_squared,
            sigmaNuNuel: (mass * mass + fieldMass * fieldMass) * c_squared * c_squared, 
            sigmaNuiNujZel: (mass * mass + fieldMass * fieldMass) * c_squared * c_squared,
            sigmaNuNuxZresNu: (mass * mass + fieldMass * fieldMass) * c_squared * c_squared, # new computations
            
            sigmaNuElGamma: (np.sqrt(mW2)+np.sqrt(me2))**2,
            sigmaNuMuGamma: (np.sqrt(mW2)+np.sqrt(mm2))**2,
            sigmaNuTauGamma: (np.sqrt(mW2)+np.sqrt(mt2))**2,
            
            sigmaNuNuxWProd: 4.*mW2,
            sigmaNuNuxZProd: 4.*mZ2,
            sigma_ZZ_incl_Rhorry: 4.*mZ2, # new computations
            sigmaNuNuxWZEl: 4.*me2,
            sigmaNuNuxWZMu: 4.*mm2,
            sigmaNuNuxWZTa: 4.*mt2,
            
            sigmaNuNuxZresEl: 4.*me2,
            sigmaNuNuxZresMu: 4.*mm2,
            sigmaNuNuxZresTa: 4.*mt2,
            sigmaNuNuxZresUp: 4.*mUp2,
            sigmaNuNuxZresDown: 4.*mDown2,
            sigmaNuNuxZresCharm: 4.*mCharm2,
            sigmaNuNuxZresStrange: 4.*mStrange2,
            sigmaNuNuxZresTop: 4.*mTop2,
            sigmaNuNuxZresBottom: 4.*mBottom2,
            
            sigmaNuiNuxjWElMux: (np.sqrt(me2) + np.sqrt(mm2))**2.,
            sigmaNuiNuxjWElTax: (np.sqrt(me2) + np.sqrt(mt2))**2.,
            sigmaNuiNuxjWMuElx: (np.sqrt(me2) + np.sqrt(mm2))**2.,
            sigmaNuiNuxjWMuTax: (np.sqrt(mt2) + np.sqrt(mm2))**2.,
            sigmaNuiNuxjWTaElx: (np.sqrt(me2) + np.sqrt(mt2))**2.,
            sigmaNuiNuxjWTaMux: (np.sqrt(mt2) + np.sqrt(mm2))**2.
             
            }[sigma]

def getEmin(sigma, field, z=0):
    """ Return minimum required cosmic ray energy for interaction *sigma* with *field* """
    return getSmin(sigma) / 4 / field.getEmax(z=z)

def getEmin_massiveBackground(sigma, mass, field, z=0):
    """ Return minimum required (massive) neutrino energy for interaction *sigma* with *field* of (massive) neutrinos"""
    
    if (field.name == 'CMB' or field.name == 'IRB_Saldana21' or field.name == 'URB_Nitu21'):   # to generalise to all the photon fields 
        return getEffectiveSmin(sigma, mass, field) / 4 / field.getEmax(z=z)
    else:
        return getEffectiveSmin(sigma, mass, field) * 0.5 / (np.sqrt(field.getPmax(z=z)**2 * c_squared + field.mass * field.mass * c_squared * c_squared) + c_light * field.getPmax(z=z))

def process(sigma, field, name, z):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the NuNu-process
        field : neutrino field as defined in neutrinoField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """
    
    folder = f'data/NeutrinoInteractionsRedshift{z:.2f}massless/{name}'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = np.logspace(4, 28, 2 ** 18 + 1) * eV**2  
    xs = getTabulatedXS(sigma, s_kin)
    
    # tabulated energies, limit to energies where the interaction is possible
    Emin = getEmin(sigma, field, z=z)
    E = np.logspace(10, 27, 1000) * eV 
    E = E[E > Emin]
    
    EmineV = Emin / eV
    print("Emin (eV): " + f"{EmineV:.2e}")
    
    rate = interactionRate.calc_rate_s(s_kin, xs, E, field, z=z)

    # save
    fname = folder + '/rate_%s.txt' % field.name
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)

    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getSmin(sigma)  # s threshold for interaction
    
    # both fields are considered relativistic
    skin2 = 4 * field.getEmin(z=z) * E[0]  # minimum achievable s in collision with background neutrino (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 28, 380000 + 1) * eV**2 
    skin = skin[skin > skin_min] 

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, z=z, cdf=True)

    print("rate shape: ", rate.shape)
    print("max/min final rate (Mpc^-1): ", np.max(rate), np.min(rate))

    # downsample
    skin_save = np.logspace(4, 28, 190 + 1) * eV**2 
    skin_save = skin_save[skin_save > skin_min] 
    rate_save = np.array([np.interp(skin_save, skin, r) for r in rate])

    # save
    data = np.c_[np.log10(E / eV), rate_save]  # prepend log10(E/eV) as first column
    row0 = np.r_[0, np.log10(skin_save / eV**2)][np.newaxis]
    data = np.r_[row0, data]  # prepend log10(s_kin/eV^2) as first row

    fname = folder + '/cdf_%s.txt' % field.name
    fmt = '%.2f' + '\t%6.5e' * np.shape(rate_save)[1]
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row" )
    except:
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row")
    np.savetxt(fname, data, fmt=fmt, header=header)

    del data, rate, skin, skin_save, rate_save
    

def process_photonBackground(sigma, mass, field, name, z=0):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the NuNu-process
        mass : mass of the propagating neutrino (kg)
        field : neutrino field as defined in neutrinoField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """
    
    folder = f'dataOff/NeutrinoInteractions/{name}/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = np.logspace(4, 28, 2 ** 18 + 1) * eV**2  
    xs = getTabulatedEffectiveXS(sigma, s_kin, mass, field)
    
    print(field)
    print(field.name)
    
    # tabulated energies, limit to energies where the interaction is possible
    Emin = getEmin_massiveBackground(sigma, mass, field, z)
    E = np.logspace(10, 27, 500) * eV 
    E = E[E > Emin]
    
    EmineV = Emin / eV
    print("Emin (eV): " + f"{EmineV:.2e}")
    
    rate = interactionRate.calc_rate_s(s_kin, xs, E, field, z=z)

    masseV = mass / eV * c_light * c_light
    tol = 1e-4

    if abs(masseV - 0) < tol:
        massNu = "m1"
    elif abs(masseV - 8.6e-3) < tol:
        massNu = "m2"
    elif abs(masseV - 50e-3) < tol:
        massNu = "m3"

    # save
    fname = folder + '/rate_%s_%s.txt' % (field.name, massNu) # _z%.1f , z)
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)

    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getEffectiveSmin(sigma, mass, field)  # s threshold for interaction
    
    # both fields are considered relativistic
    skin2 = 4 * field.getEmin(z=z) * E[0]  # minimum achievable s in collision with background photon (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 28, 380000 + 1) * eV**2 
    skin = skin[skin > skin_min] 

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, z=z, cdf=True)

    print("rate shape: ", rate.shape)
    print("max/min final rate (Mpc^-1): ", np.max(rate), np.min(rate))

    # downsample
    skin_save = np.logspace(4, 28, 390 + 1) * eV**2 
    skin_save = skin_save[skin_save > skin_min] 
    rate_save = np.array([np.interp(skin_save, skin, r) for r in rate])

    # save
    data = np.c_[np.log10(E / eV), rate_save]  # prepend log10(E/eV) as first column
    row0 = np.r_[0, np.log10(skin_save / eV**2)][np.newaxis]
    data = np.r_[row0, data]  # prepend log10(s_kin/eV^2) as first row

    fname = folder + '/cdf_%s_%s.txt' % (field.name, massNu) #_z%.1f , z)
    fmt = '%.2f' + '\t%6.5e' * np.shape(rate_save)[1]
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row" )
    except:
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row")
    np.savetxt(fname, data, fmt=fmt, header=header)

    del data, rate, skin, skin_save, rate_save


def getEmin_massiveBackground_fromFile(sigmaID, mass, field, z=0):
    """ Return minimum required (massive) neutrino energy for interaction *sigma* with *field* of (massive) neutrinos"""
    return {
        29: (np.sqrt(mW2)+np.sqrt(mm2))**2 / 4 / field.getEmax(z=z),
        113: (np.sqrt(mt2)+np.sqrt(mm2))**2 / 4 / field.getEmax(z=z)
    } [sigmaID]

def process_photonBackground_fromFile(sigmaID, mass, field, name, z=0):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the NuNu-process
        mass : mass of the propagating neutrino (kg)
        field : neutrino field as defined in neutrinoField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """
    
    folder = "/Users/a39392/Desktop/neutrinoGammaInteraction/NuPropa/checks/"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    data = np.genfromtxt(
        folder + f"SigmaIncl_Ecms_channel{sigmaID}_s1.txt",  # replace with your filename
        comments="#",         # ignore lines starting with #
        usecols=(0, 1)        # first and third columns
    )
    
    pb_to_m2 = 1e-40  
    
    # Split into separate arrays
    s = data[:, 0]**2 * GeV**2   # GeV
    xs = data[:, 1] * pb_to_m2 # pb to m2 
    
    from scipy.interpolate import interp1d
    
    # Kinematic points for interpolation
    s_kin = np.logspace(4, 28, 2**18 + 1) * eV**2
    s_points = s_kin + mass**2 * c_squared**2
    
    # Create interpolation function
    interp_func = interp1d(
        s, xs, kind='linear', fill_value='extrapolate', assume_sorted=True
    )
    
    # Interpolate
    xs_ext = interp_func(s_points)
    
    print(field)
    print(field.name)
    
    # tabulated energies, limit to energies where the interaction is possible
    Emin = getEmin_massiveBackground_fromFile(sigmaID, mass, field, z)
    E = np.logspace(10, 27, 500) * eV 
    E = E[E > Emin]
    
    EmineV = Emin / eV
    print("Emin (eV): " + f"{EmineV:.2e}")
    
    rate = interactionRate.calc_rate_s(s_kin, xs_ext, E, field, z=z)

    masseV = mass / eV * c_light * c_light
    tol = 1e-4

    if abs(masseV - 0) < tol:
        massNu = "m1"
    elif abs(masseV - 8.6e-3) < tol:
        massNu = "m2"
    elif abs(masseV - 50e-3) < tol:
        massNu = "m3"

    # save
    fname = folder + '/rate_%s_%s_%i.txt' % (field.name, massNu, sigmaID) # _z%.1f , z)
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)
    
    '''
    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getEffectiveSmin(sigma, mass, field)  # s threshold for interaction
    
    # both fields are considered relativistic
    skin2 = 4 * field.getEmin(z=z) * E[0]  # minimum achievable s in collision with background photon (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 28, 380000 + 1) * eV**2 
    skin = skin[skin > skin_min] 

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, z=z, cdf=True)

    print("rate shape: ", rate.shape)
    print("max/min final rate (Mpc^-1): ", np.max(rate), np.min(rate))

    # downsample
    skin_save = np.logspace(4, 28, 390 + 1) * eV**2 
    skin_save = skin_save[skin_save > skin_min] 
    rate_save = np.array([np.interp(skin_save, skin, r) for r in rate])

    # save
    data = np.c_[np.log10(E / eV), rate_save]  # prepend log10(E/eV) as first column
    row0 = np.r_[0, np.log10(skin_save / eV**2)][np.newaxis]
    data = np.r_[row0, data]  # prepend log10(s_kin/eV^2) as first row

    fname = folder + '/cdf_%s_%s.txt' % (field.name, massNu) #_z%.1f , z)
    fmt = '%.2f' + '\t%6.5e' * np.shape(rate_save)[1]
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row" )
    except:
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row")
    np.savetxt(fname, data, fmt=fmt, header=header)
    
    del data, rate, skin, skin_save, rate_save
    '''


def process_massiveBackground(sigma, mass, field, name, z):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the NuNu-process
        mass : mass of the propagating neutrino (in kg!)
        field : neutrino field as defined in neutrinoField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """
    
    folder = f'dataOff/NeutrinoInteractions/{name}/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = np.logspace(4, 28, 2 ** 18 + 1) * eV**2  
    xs = getTabulatedEffectiveXS(sigma, s_kin, mass, field)
    '''
    import matplotlib.pyplot as plt 
    plt.plot(s_kin / eV**2, xs / cm**2)

    plt.xlabel(r"$s_{kin}$ ($eV^{2}$)")
    plt.ylabel(r"$\sigma$ ($cm^{2}$)")
    
    dirFig = "/Users/a39392/Desktop/neutrinoGammaInteraction/images/newComputationXS/"
    figName = name.split('/', 1)[1]
    plt.loglog()
    ext = ".png"
    
    target_s = 4e22 * eV**2
    cross_at_target = np.interp(target_s, s_kin, xs)
    print("The XS at s=", target_s / eV**2, " eV2 is ", cross_at_target / cm**2, " cm2")
    
    plt.savefig(dirFig + figName + ext)
    plt.show()
    '''
    # tabulated energies, limit to energies where the interaction is possible
    Emin = getEmin_massiveBackground(sigma, mass, field, z=z)    
    EmineV = Emin / eV
    
    E = np.logspace(10, 27, 500) * eV  # to have higher resolution at the peak 
    E = E[E > Emin]
    
    rate = interactionRate.calc_rate_s_momenta(s_kin, xs, E, mass, field, z=z)

    masseV = mass / eV * c_light * c_light
    tol = 1e-4

    if abs(masseV - 0) < tol:
        massNu = "m1"
    elif abs(masseV - 8.6e-3) < tol:
        massNu = "m2"
    elif abs(masseV - 50e-3) < tol:
        massNu = "m3"
        
    # save
    fname = folder + '/rate_%s_%s_z%.1f.txt' % (field.name, massNu, z) 
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nneutrino field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)

    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getEffectiveSmin(sigma, mass, field)  # s threshold for interaction
    print("skin1 (eV**2): ", skin1 / eV**2)
    # both fields are considered relativistic
    # minimum achievable s in collision with background neutrino (at any tabulated E), setting the interaction angle to pi
    skin2 = 2 * E[0] * (np.sqrt((field.getPmin() * c_light) ** 2 + (field.mass * c_squared) ** 2) + field.getPmin() * c_light)  
    print("skin2 (eV**2): ", skin2 / eV**2)
    skin_min = max(skin1, skin2)
    print("skin_min (eV**2): ", skin_min / eV**2)
    
    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 28, 380000 + 1) * eV**2 
    skin = skin[skin > skin_min] 

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s_momenta(skin, xs, E, mass, field, z=z, cdf=True)

    # downsample
    skin_save = np.logspace(4, 28, 390 + 1) * eV**2 
    skin_save = skin_save[skin_save > skin_min] 
    rate_save = np.array([np.interp(skin_save, skin, r) for r in rate])

    # save
    data = np.c_[np.log10(E / eV), rate_save]  # prepend log10(E/eV) as first column
    row0 = np.r_[0, np.log10(skin_save / eV**2)][np.newaxis]
    data = np.r_[row0, data]  # prepend log10(s_kin/eV^2) as first row

    fname = folder + '/cdf_%s_%s_z%.1f.txt' % (field.name, massNu, z) 
    fmt = '%.2f' + '\t%6.5e' * np.shape(rate_save)[1]
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row" )
    except:
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row")
    np.savetxt(fname, data, fmt=fmt, header=header)

    del data, rate, skin, skin_save, rate_save

'''
if __name__ == "__main__":
    
    for field in fields_CnuB:
        print(field.name)
        
        process(sigmaNuNuel, field, 'NeutrinoNeutrinoInteraction/NeutrinoNeutrinoElastic', z)
        process(sigmaNuiNujZel, field, 'NeutrinoNeutrinoInteraction/NeutrinoiNeutrinojElastic', z)
        process(sigmaNuNuxWProd, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoWProduction', z)
        process(sigmaNuNuxZProd, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoZProduction', z)
        process(sigmaNuNuxZel, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoElastic', z)
        process(sigmaNuiNuxjZel, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojElastic', z)
        
        process(sigmaNuNuxWZEl, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoElectron', z)
        process(sigmaNuNuxWZMu, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoMuon', z)
        process(sigmaNuNuxWZTa, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoTau', z)
        
        process(sigmaNuNuxZresEl, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceElectron', z)
        process(sigmaNuNuxZresMu, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceMuon', z)
        process(sigmaNuNuxZresTa, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceTau', z)
        process(sigmaNuNuxZresUp, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceUp', z)
        
        process(sigmaNuNuxZresDown, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceDown', z)
        process(sigmaNuNuxZresCharm, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceCharm', z)
        process(sigmaNuNuxZresStrange, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceStrange', z)
        process(sigmaNuNuxZresTop, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceTop', z)
        process(sigmaNuNuxZresBottom, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceBottom', z)
        
        process(sigmaNuiNuxjWElMux, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojElectronAntimuon', z)
        process(sigmaNuiNuxjWElTax, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojElectronAntitau', z)
        process(sigmaNuiNuxjWMuElx, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojMuonAntielectron', z)
        process(sigmaNuiNuxjWMuTax, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojMuonAntitau', z)
        process(sigmaNuiNuxjWTaElx, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojTauAntielectron', z)
        process(sigmaNuiNuxjWTaMux, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojTauAntimuon', z)
'''    

masses = np.array([0, 8.6, 50]) * 1e-3 * eV / c_light / c_light
redshifts = np.array([0, 2, 5, 8, 11, 15, 20, 25, 30, 40, 50])
sigmaID = [113, 29]

# the CMB does not change with the redshift, naive scaling of the field and the IMFP
if __name__ == "__main__":

    for field in cmb: #reduced_fields:
        for mass in masses:
            for ID in sigmaID: 
                process_photonBackground_fromFile(ID, mass, field, ID)
            '''
            print(field.name)
            process_photonBackground(sigmaNuElGamma, mass, field, 'NeutrinoPhotonInteraction/NeutrinoElectronPhotonInteraction')
            process_photonBackground(sigmaNuMuGamma, mass, field, 'NeutrinoPhotonInteraction/NeutrinoMuonPhotonInteraction')
            process_photonBackground(sigmaNuTauGamma, mass, field, 'NeutrinoPhotonInteraction/NeutrinoTauPhotonInteraction')
            '''

if __name__ == "__main__":
    
    for z in redshifts:
        for field in fields_massiveCnuB:   
            for mass in masses:
            
                
                '''
                print(field.name)
                process_massiveBackground(sigmaNuNuel, mass, field, 'NeutrinoNeutrinoInteraction/NeutrinoNeutrinoElastic', z)
                process_massiveBackground(sigmaNuiNujZel, mass, field, 'NeutrinoNeutrinoInteraction/NeutrinoiNeutrinojElastic', z)
             
                
                # done
                #process_massiveBackground(sigmaNuNuxWProd, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoWProduction', z)    
                process_massiveBackground(sigmaNuNuxZProd, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoZProduction', z)
                process_massiveBackground(sigmaNuNuxZel, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoElastic', z)
            
                # done
                #process_massiveBackground(sigmaNuiNuxjZel, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojElastic', z)
                
                process_massiveBackground(sigmaNuNuxWZEl, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoElectron', z)
                process_massiveBackground(sigmaNuNuxWZMu, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoMuon', z)
                process_massiveBackground(sigmaNuNuxWZTa, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoTau', z)
                
                # done
                #process_massiveBackground(sigmaNuNuxZresEl, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceElectron', z)
                
                process_massiveBackground(sigmaNuNuxZresMu, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceMuon', z)
                
                process_massiveBackground(sigmaNuNuxZresTa, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceTau', z)
                process_massiveBackground(sigmaNuNuxZresUp, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceUp', z) 
                process_massiveBackground(sigmaNuNuxZresDown, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceDown', z)
                process_massiveBackground(sigmaNuNuxZresCharm, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceCharm', z)
                process_massiveBackground(sigmaNuNuxZresStrange, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceStrange', z)
                process_massiveBackground(sigmaNuNuxZresTop, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceTop', z)
                process_massiveBackground(sigmaNuNuxZresBottom, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceBottom', z)
                
                process_massiveBackground(sigmaNuiNuxjWElMux, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojElectronAntimuon', z)
                process_massiveBackground(sigmaNuiNuxjWElTax, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojElectronAntitau', z)
                process_massiveBackground(sigmaNuiNuxjWMuElx, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojMuonAntielectron', z)
                process_massiveBackground(sigmaNuiNuxjWMuTax, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojMuonAntitau', z)
                process_massiveBackground(sigmaNuiNuxjWTaElx, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojTauAntielectron', z)
                
                #process_massiveBackground(sigmaNuiNuxjWTaMux, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoiAntineutrinojTauAntimuon', z)
                
                #process_massiveBackground(sigmaNuNuxZresNu, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoResonanceNu', z)
 
                # sigma_ZZ_incl_Rhorry
                process_massiveBackground(sigma_ZZ_incl_Rhorry, mass, field, 'NeutrinoAntineutrinoInteraction/NeutrinoAntineutrinoZProduction', z)
                #break
                '''
            
    
        
        
        