from __future__ import division
import numpy as np
import interactionRate
import os
import gitHelp as gh
import math
from crpropa import eV, mass_electron, c_light, h_planck, GeV
from calc_all import fields_cmbebl, fields_urb, fields_CnuB
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
#def sigmaTrident(s): # to compute the total cross section for these
#    return 0.

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
    smin = 4.*mNu2
    if (s < smin):
        return 0.
    
    sthr = 5e15 * eV * eV
    F1val = F1exp0(mZ2, s, sthr)
    return 0.5*Gf*Gf/math.pi/h_planck/h_planck/c_light*(2.*math.pi)*(2.*math.pi)/c_light*s*F1val 
    
def sigmaNuNuxZel(s):
    """Neutrino-(anti)neutrino interaction Z-exchange interaction cross section, see Roulet (1992) Eq. 2.3"""
    
    smin = mNu2
    sthr = mZ2
    if (s < smin or s > sthr):
        return 0.
    
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
    
    smin = mZ2
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
    
def sigmaNuNuxLepZ(s):
    """Neutrino-(anti)neutrino (t-channel lepton exchange + s-channel Z-exchange) interaction cross section, see Roulet (1992) Eq. 2.6"""
    smin = 4.*mW2
    if (s < smin):
        return 0.
    
    beta = np.sqrt(1.-4.*mW2/s)
    L = np.log((1.+beta)/(1.-beta))
    y = s/mW2
    
    return Gf*Gf/12./math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*s*beta*(beta*beta*mW2*mW2/(s-mZ2)**2*(12.+20.*y+y*y)+2.*mW2/(s-mW2)/y/y*(24.+28.*y-18.*y*y-y**3+48./beta/y*(1.+2.*y)*L)+(y*y+20.*y-48.-48./beta/y*L*(2.-y))/y/y)

def sigmaNuiNujZel(s):
    """(anti)Neutrino_i-(anti)neutrino_j Z-exchange t-channel elastic scatter cross section, see Roulet (1992) Eq. 2.7"""
    smin = 4.*mNu2
    if (s < smin):
        return 0.
    
    return Gf*Gf*mZ2*0.5/math.pi/h_planck/h_planck/c_light/c_light/(2.*math.pi)/(2.*math.pi)*s/(s+mZ2)

def sigmaNuNuel(s):
    """(anti)Neutrino-(anti)neutrino u-channel elastic scatter cross section, see Roulet (1992) Eq. 2.8"""
    smin = 4.*mNu2
    if (s < smin):
        return 0.
    
    return Gf*Gf*mZ2/math.pi/h_planck/h_planck/c_light/c_light*(2.*math.pi)*(2.*math.pi)*(s/(s+mZ2)+2.*mZ2/(2.*mZ2+s)*np.log(1.+s/mZ2))

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
    if sigma in (sigmaNuNuel, sigmaNuNuxLepZ, sigmaNuNuxWZEl,
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
                 sigmaNuiNuxjWTaMux, sigmaNuiNuxjZel):  
            # neutrino-neutrino interaction
        return np.array([sigma(s) for s in skin + 4.*mNu2 ])
    return False

def getSmin(sigma):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction """

    return {
            sigmaNuElGamma: (np.sqrt(mW2)+np.sqrt(me2))**2,
            sigmaNuMuGamma: (np.sqrt(mW2)+np.sqrt(mm2))**2,
            sigmaNuTauGamma: (np.sqrt(mW2)+np.sqrt(mt2))**2,
            
            sigmaNuNuel: 4.*mNu2, 
            sigmaNuiNujZel: 4.*mNu2, 
            sigmaNuNuxLepZ: 4.*mW2,
            sigmaNuNuxZel: 4.*mNu2,
            sigmaNuiNuxjZel: 4.*mNu2,
            
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

def getEmin(sigma, field, s_kin):
    """ Return minimum required cosmic ray energy for interaction *sigma* with *field* """
    return getSmin(sigma) / 4 / field.getEmax()

def process(sigma, field, name):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the NuNu-process
        field : neutrino field as defined in neutrinoField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """
    
    folder = 'data/NeutrinoNeutrinoInteractionsE1027Mass01/' + name
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
    Emin = getEmin(sigma, field, s_kin)
    E = np.logspace(10, 27, 281) * eV 
    E = E[E > Emin]
    
    rate = interactionRate.calc_rate_s(s_kin, xs, E, field)

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
    skin2 = 4 * field.getEmin() * E[0]  # minimum achievable s in collision with background neutrino (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 28, 380000 + 1) * eV**2 
    skin = skin[skin > skin_min]

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, cdf=True)

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

if __name__ == "__main__":

    for field in fields_cmbebl+fields_urb:
        print(field.name)
        process(sigmaNuElGamma, field, 'NeutrinoElectronPhotonInteraction')
        process(sigmaNuMuGamma, field, 'NeutrinoMuonPhotonInteraction')
        process(sigmaNuTauGamma, field, 'NeutrinoTauPhotonInteraction')
    
    for field in fields_CnuB:
        print(field.name)
        process(sigmaNuNuel, field, 'NeutrinoNeutrinoElastic')
        process(sigmaNuiNujZel, field, 'NeutrinoiNeutrinojElastic')
        process(sigmaNuNuxLepZ, field, 'NeutrinoAntineutrinoWProduction')
        process(sigmaNuNuxZel, field, 'NeutrinoAntineutrinoElastic')
        process(sigmaNuiNuxjZel, field, 'NeutrinoiAntineutrinojElastic')
        
        process(sigmaNuNuxWZEl, field, 'NeutrinoAntineutrinoElectron')
        process(sigmaNuNuxWZMu, field, 'NeutrinoAntineutrinoMuon')
        process(sigmaNuNuxWZTa, field, 'NeutrinoAntineutrinoTau')
        
        process(sigmaNuNuxZresEl, field, 'NeutrinoAntineutrinoResonanceElectron')
        process(sigmaNuNuxZresMu, field, 'NeutrinoAntineutrinoResonanceMuon')
        process(sigmaNuNuxZresTa, field, 'NeutrinoAntineutrinoResonanceTau')
        process(sigmaNuNuxZresUp, field, 'NeutrinoAntineutrinoResonanceUp')
        process(sigmaNuNuxZresDown, field, 'NeutrinoAntineutrinoResonanceDown')
        process(sigmaNuNuxZresCharm, field, 'NeutrinoAntineutrinoResonanceCharm')
        process(sigmaNuNuxZresStrange, field, 'NeutrinoAntineutrinoResonanceStrange')
        process(sigmaNuNuxZresTop, field, 'NeutrinoAntineutrinoResonanceTop')
        process(sigmaNuNuxZresBottom, field, 'NeutrinoAntineutrinoResonanceBottom')
        
        process(sigmaNuiNuxjWElMux, field, 'NeutrinoiAntineutrinojElectronAntimuon')
        process(sigmaNuiNuxjWElTax, field, 'NeutrinoiAntineutrinojElectronAntitau')
        process(sigmaNuiNuxjWMuElx, field, 'NeutrinoiAntineutrinojMuonAntielectron')
        process(sigmaNuiNuxjWMuTax, field, 'NeutrinoiAntineutrinojMuonAntitau')
        process(sigmaNuiNuxjWTaElx, field, 'NeutrinoiAntineutrinojTauAntielectron')
        process(sigmaNuiNuxjWTaMux, field, 'NeutrinoiAntineutrinojTauAntimuon')
        
        
        
        
        
        