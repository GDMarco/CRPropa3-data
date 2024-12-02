from __future__ import division
import numpy as np
import interactionRate
import os
import gitHelp as gh
from calc_all import fields_cmbebl, fields_urb
from units import eV, mass_electron, c_light, sigma_thomson, alpha_finestructure, h_planck, second
from units import cm, mass_muon, sigma_thomson_muon, sigma_thomson_tauon, mass_charged_pion, mass_tauon, mass_charged_kaon, mass_neutral_kaon, mass_neutral_pion
from units import widthNP
#import matplotlib.pyplot as plt

me2 = (mass_electron*c_light**2.) ** 2  # squared electron mass [J^2/c^4]
mm2 = (mass_muon*c_light**2.) ** 2  # squared muon mass [J^2/c^4]
mp2 = (mass_charged_pion*c_light**2.) ** 2 # squared charged pion mass [J^2/c^4]
mt2 = (mass_tauon*c_light**2) ** 2 # squared tauon mass [J^2/c^4]
mck2 = (mass_charged_kaon*c_light**2) ** 2
mnp2 = (mass_neutral_pion*c_light**2) ** 2 
mnk2 = (mass_neutral_kaon*c_light**2) ** 2

def pbTom2(CSpb):
    pico = 1e-12
    frombtom2 = 1e-28
    #fromm2tocm2 = 1e4 
    
    CSb = CSpb * pico
    CSm2 = CSb * frombtom2
    #CScm2 = CSm2 * fromm2tocm2
    
    return CSm2
    
def sigmaPP(s):
    """ Pair production cross section (Breit-Wheeler), see Lee 1996 """
    smin = 4 * me2
    if (s < smin):
        return 0.
    
    b = np.sqrt(1 - smin / s)
    return sigma_thomson * 3 / 16 * (1 - b**2) * ((3 - b**4) * (np.log1p(b) - np.log1p(-b)) - 2 * b * (2 - b**2))

def sigmaMPP(s):
    """ Muon Pair production cross section (Breit-Wheeler), see Lee 1996 """
    smin = 4 * mm2
    if (s < smin):
        return 0.

    b = np.sqrt(1 - smin / s)
    return sigma_thomson_muon * 3 / 16 * (1 - b**2) * ((3 - b**4) * (np.log1p(b) - np.log1p(-b)) - 2 * b * (2 - b**2))

def sigmaTauPP(s):
    """ Tauon Pair production cross section (Breit-Wheeler), see Lee 1996 """
    smin = 4 * mt2
    if (s < smin):
        return 0.

    b = np.sqrt(1 - smin / s)
    return sigma_thomson_tauon * 3 / 16 * (1 - b**2) * ((3 - b**4) * (np.log1p(b) - np.log1p(-b)) - 2 * b * (2 - b**2))

def sigmaCPPP(s):
    """ Charged Pion Pair production cross section, see Brodsky+ 1971 (Born approximation, for point-like pion). """
    smin = 4 * mp2
    if (s < smin):
        return 0.
    
    # I multiply the cross section by h_planck**2 * c_light**2 / (2.*np.pi)**2 to move to the SI units
    C = 2 * np.pi * alpha_finestructure*alpha_finestructure * c_light*c_light * h_planck*h_planck / (2.*np.pi) / (2.*np.pi) / s
    y = smin / s
    
    return C * ((1 + y) * np.sqrt(1 - y) - 2 * y * (1 - y / 2) * np.log(np.sqrt(1 / y) + np.sqrt(1 / y - 1)))

def sigmaCKPP(s):
    """ Charged Kaon Pair production cross section, see Brodsky+ 1971 (Born approximation, for point-like pion) & 2A. I. Akhiezer and V. B.Berestetskii, Quantum Electrodynamics (Interscience, New York, 1965), p. 844. 
    This expression is smaller by a factor of 2 than that in the book reference [Eq. (60.7), p.844]"""
    smin = 4 * mck2
    if (s < smin):
        return 0.
    
    # I multiply the cross section by h_planck**2 * c_light**2 / (2.*np.pi)**2 to move to the SI units
    C = 2 * np.pi * alpha_finestructure*alpha_finestructure * c_light*c_light * h_planck*h_planck / (2.*np.pi) / (2.*np.pi) / s
    y = smin / s
    
    return C * ((1 + y) * np.sqrt(1 - y) - 2 * y * (1 - y / 2) * np.log(np.sqrt(1 / y) + np.sqrt(1 / y - 1))) 

'''
#the two cross sections for neutral pion and kaon are not correct, they cannot be computed in the Born approximation!

def sigmaNKPP(s):
    """ Neutral Kaon Pair production cross section, see Brodsky+ 1971 (Born approximation, for point-like pion). """
    """ Not sure it works!! Maybe they comes from strong interactions"""
    smin = 4 * mnk2
    if (s < smin):
        return 0.
    
    # I multiply the cross section by h_planck**2 * c_light**2 / (2.*np.pi)**2 to move to the SI units
    C = 2 * np.pi * alpha_finestructure*alpha_finestructure * c_light*c_light * h_planck*h_planck / (2.*np.pi) / (2.*np.pi) / s
    y = smin / s
    
    return C * ((1 + y) * np.sqrt(1 - y) - 2 * y * (1 - y / 2) * np.log(np.sqrt(1 / y) + np.sqrt(1 / y - 1)))

def sigmaNPPP(s):
    """ Neutral Pion Pair production cross section, see Brodsky+ 1971 (Born approximation, for point-like pion). """
    """Not sure it works!! Maybe they comes from strong interactions"""
    smin = 4 * mnp2
    if (s < smin):
        return 0.
    
    # I multiply the cross section by h_planck**2 * c_light**2 / (2.*np.pi)**2 to move to the SI units
    C = 2 * np.pi * alpha_finestructure*alpha_finestructure * c_light*c_light * h_planck*h_planck / (2.*np.pi) / (2.*np.pi) / s
    y = smin / s
    
    return C * ((1 + y) * np.sqrt(1 - y) - 2 * y * (1 - y / 2) * np.log(np.sqrt(1 / y) + np.sqrt(1 / y - 1)))
'''

def sigmaNPP(s): # still to implement!
    """ narrow resonant neutral Pion production cross section, see Brodsky+ 1971 (from QED Lagrangian). 
         (delta function approximated as a gaussian with width equals to the decay ones)   
    """
   
    # check the units!
    return 8 * np.pi * np.pi * c_light*c_light * h_planck*h_planck / (2.*np.pi) / (2.*np.pi) / np.sqrt(mnp2) / np.sqrt(2 * np.pi) / widthNP * np.exp(-(mnp2 - s)**2 / 2 / widthNP ** 4)

def sigmaDPP(s):
    """ Double-pair production cross section, see R.W. Brown eq. (4.5) with k^2 = q^2 = 0 """
    smin = 16 * me2
    if (s < smin):
        return 0

    return 6.45E-34 * (1 - smin / s)**6

def sigmaICS(s):
    """ Inverse Compton scattering cross sections, see Lee 1996 """
    smin = me2
    if (s < smin):  # numerically unstable close to smin
        return 0

    # note: formula unstable for (s - smin) / smin < 1E-5
    b = (s - smin) / (s + smin)
    A = 2 / b / (1 + b) * (2 + 2 * b - b**2 - 2 * b**3)
    B = (2 - 3 * b**2 - b**3) / b**2 * (np.log1p(b) - np.log1p(-b))
    return sigma_thomson * 3 / 8 * smin / s / b * (A - B)


def sigmaTPP(s):
    """ Triplet-pair production cross section, see Lee 1996 """
    beta = 28 / 9 * np.log(s / me2) - 218 / 27
    if beta < 0:
        return 0
    
    return sigma_thomson * 3 / 8 / np.pi * alpha_finestructure * beta

def sigmaEMPP(s):
    """Electron muon pair production, total cross section from tables found in MUNHECA code"""
    smin = (np.sqrt(me2) + 2. * np.sqrt(mm2)) ** 2 
    if (s < smin):
        return 0
    
    dataPath = "/Applications/CRPropa/EMCascadePlugins/CRPropa3-data/data/EMElectronMuonPairProduction/"
    filename = "EMPP_totalCS.txt"
    
    sTab, CSTab = np.genfromtxt(dataPath + filename, comments='#', usecols=(0,1), unpack=True) 
    CSTab = pbTom2(CSTab)
    sTab = sTab * eV**2
    
    if ((s < sTab[0]) or (s > sTab[-1])):
        return 0 
    
    from scipy.interpolate import interp1d 
    
    interpFunc = interp1d(sTab, CSTab)
    CS = interpFunc(s)
    
    return CS

def getTabulatedXS(sigma, skin):
    """ Get crosssection for tabulated s_kin """
    if sigma in (sigmaPP, sigmaDPP, sigmaMPP, sigmaCPPP, sigmaTauPP, sigmaCKPP):  # photon interactions
        return np.array([sigma(s) for s in skin])
    if sigma in (sigmaTPP, sigmaICS, sigmaEMPP):  # electron interactions
        return np.array([sigma(s) for s in skin + me2])
    return False

def getSmin(sigma):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction """
    return {sigmaPP: 4 * me2,
            sigmaMPP: 4 * mm2,
            sigmaTauPP: 4 * mt2,
            sigmaCPPP: 4 * mp2,
            sigmaDPP: 16 * me2,
            sigmaTPP: np.exp((218 / 27) / (28 / 9)) * me2 - me2,
            sigmaICS: 1e-40 * me2, 
            sigmaEMPP: (np.sqrt(me2) + 2. * np.sqrt(mm2)) ** 2, 
            sigmaCKPP: 4 * mck2
            }[sigma]


def getEmin(sigma, field):
    """ Return minimum required cosmic ray energy for interaction *sigma* with *field* """
    return getSmin(sigma) / 4 / field.getEmax()


def process(sigma, field, name):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the EM-process
        field : photon field as defined in photonField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """

    # output folder
    folder = 'data/' + name
    if not os.path.exists(folder):
        os.makedirs(folder)

    # tabulated energies, limit to energies where the interaction is possible
    Emin = getEmin(sigma, field)
    E = np.logspace(8, 26, 281) * eV
    E = E[E > Emin]
    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = np.logspace(4, 26, 2 ** 18 + 1) * eV**2 # smax was 23
    xs = getTabulatedXS(sigma, s_kin)
    rate = interactionRate.calc_rate_s(s_kin, xs, E, field)

    # save
    fname = folder + '/rate_%s.txt' % field.name
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)

    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getSmin(sigma)  # s threshold for interaction
    skin2 = 4 * field.getEmin() * E[0]  # minimum achievable s in collision with background photon (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = np.logspace(4, 26, 380000 + 1) * eV**2 # smax was 23
    skin = skin[skin > skin_min]

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, cdf=True)

    # downsample
    skin_save = np.logspace(4, 26, 190 + 1) * eV**2 # smax was 23
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
        process(sigmaPP, field, 'EMPairProduction')
        process(sigmaMPP, field, 'EMMuonPairProduction')
        process(sigmaTauPP, field, 'EMTauonPairProduction')
        process(sigmaCPPP, field, 'EMChargedPionPairProduction')
        process(sigmaCKPP, field, 'EMChargedKaonPairProduction')
        process(sigmaEMPP, field, 'EMElectronMuonPairProduction')
        process(sigmaDPP, field, 'EMDoublePairProduction')
        process(sigmaTPP, field, 'EMTripletPairProduction')
        process(sigmaICS, field, 'EMInverseComptonScattering')
