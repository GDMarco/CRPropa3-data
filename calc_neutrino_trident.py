from __future__ import division
import numpy as np
import interactionRate
import os
import gitHelp as gh
import math
from scipy.interpolate import interp1d
from crpropa import eV, mass_electron, c_light, h_planck, GeV, cm 
from calc_all import reduced_fields
from units import me2, sigmaThomson, alpha, mm2, mt2, mW2, mZ2, mUp2, mDown2
from units import mCharm2, mStrange2, mBottom2, mTop2, mNu2, Gf, gammaZ2, sW2

# the second interacting particle is always a photon
# numbers correspond to a certain process, according to Rhorry's nomenclature (https://github.com/GDMarco/NuPropa/blob/main/PartonicCalculation/sigmaNu/Ecms_scan/README_CHANNELS)

def pbTocm2(CSpb):
    pico = 1e-12
    frombtom2 = 1e-28
    fromm2tocm2 = 1e4 
    
    CSb = CSpb * pico
    CSm2 = CSb * frombtom2
    CScm2 = CSm2 * fromm2tocm2
    
    return CScm2

def GeVToeV(Ecms):
    return Ecms * 1e9

def getTables(dataPath, processFile):
    interactionData = np.genfromtxt(dataPath + processFile, comments='#', usecols=(0, 1))
    Ecms = interactionData[:,0]
    Sigma = interactionData[:,1]
    
    Ecms2 = []
    Sigmacm2 = []
    
    for i in range(len(Ecms)):
        Ecms2.append(GeVToeV(Ecms[i]) * GeVToeV(Ecms[i]))
        Sigmacm2.append(pbTocm2(Sigma[i]))
    
    Ecms2 = np.array(Ecms2)
    Sigmacm2 = np.array(Sigmacm2)

    return Ecms2 * eV * eV, Sigmacm2 * cm * cm 

dataPath = '/Applications/CRPropa/NuPropa/PartonicCalculation/sigmaNu/Ecms_scan/'

def sigma101(s):
    '''nu gamma -> nu e- e+'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*me2))**2 
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel101_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma102(s):
    '''nu gamma -> nu mu- mu+'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel102_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma103(s):
    '''nu gamma -> nu tau- tau+'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mt2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel103_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma104(s):
    '''nu gamma -> nu d d~'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mDown2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel104_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma105(s):
    '''nu gamma -> nu u u~'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mUp2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel105_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma106(s):
    '''nu gamma -> nu s s~'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mStrange2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel106_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma107(s):
    '''nu gamma -> nu c c~'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mCharm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel107_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma108(s):
    '''nu gamma -> nu b b~'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mBottom2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel108_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma109(s):
    '''nu gamma -> nu t t~'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mTop2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel109_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS

def sigma110(s):
    '''nu_e gamma -> e- nu_mu mu+'''
    smin = (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel110_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS  

def sigma111(s):
    '''nu_e gamma -> e- nu_tau tau+'''
    smin = (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mt2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel111_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma112(s):
    '''nu_mu gamma -> mu- nu_e e+'''
    smin = (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel112_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma113(s):
    '''nu_mu gamma -> mu- nu_tau tau+'''
    smin = (np.sqrt(mNu2) + np.sqrt(mt2) + np.sqrt(mm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel113_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma114(s):
    '''nu_tau gamma -> tau- nu_e e+'''
    smin = (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mt2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel114_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma115(s):
    '''nu_tau gamma -> tau- nu_mu mu+'''
    smin = (np.sqrt(mNu2) + np.sqrt(mt2) + np.sqrt(mm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel115_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma116(s):
    '''nu_e gamma -> e- nu_e e+'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*me2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel116_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma117(s):
    '''nu_mu gamma -> mu- nu_mu mu+'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mm2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel117_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma118(s):
    '''nu_tau gamma -> tau- nu_tau tau+'''
    smin = (np.sqrt(mNu2) + np.sqrt(4*mt2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel118_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma119(s):
    '''nu_e gamma -> e- u d~'''
    smin = (np.sqrt(mUp2) + np.sqrt(me2) + np.sqrt(mDown2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel119_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma120(s):
    '''nu_mu gamma -> mu- u d~'''
    smin = (np.sqrt(mUp2) + np.sqrt(mm2) + np.sqrt(mDown2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel120_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

def sigma121(s):
    '''nu_tau gamma -> tau- u d~'''
    smin = (np.sqrt(mUp2) + np.sqrt(mt2) + np.sqrt(mDown2))**2
    if (s < smin):
        return 0.
    
    processFile = 'SigmaIncl_Ecms_channel121_s3.txt'
    Ecms2, sigma = getTables(dataPath, processFile)
    
    interpFunc = interp1d(Ecms2, sigma)
    CS = interpFunc(s)
    
    return CS 

#########################################

def getTabulatedXS(sigma, skin):
    """ Get cross section for tabulated s_kin """
    if sigma in (sigma101, sigma102, sigma103, sigma104, sigma105, sigma106, sigma107, sigma108, \
                 sigma109, sigma110, sigma111, sigma112, sigma113, sigma114, sigma115, sigma116, \
                 sigma117, sigma118, sigma119, sigma120, sigma121):
        # photon-neutrino interaction
        return np.array([sigma(s) for s in skin + mNu2])  
    return False

def getSmin(sigma):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction """

    return {
            sigma101: (np.sqrt(mNu2) + np.sqrt(4*me2))**2,
            sigma102: (np.sqrt(mNu2) + np.sqrt(4*mm2))**2,
            sigma103: (np.sqrt(mNu2) + np.sqrt(4*mt2))**2,
            sigma104: (np.sqrt(mNu2) + np.sqrt(4*mDown2))**2,
            sigma105: (np.sqrt(mNu2) + np.sqrt(4*mUp2))**2,
            sigma106: (np.sqrt(mNu2) + np.sqrt(4*mStrange2))**2,
            sigma107: (np.sqrt(mNu2) + np.sqrt(4*mCharm2))**2,
            sigma108: (np.sqrt(mNu2) + np.sqrt(4*mBottom2))**2,
            sigma109: (np.sqrt(mNu2) + np.sqrt(4*mTop2))**2,
            sigma110: (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mm2))**2,
            sigma111: (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mt2))**2,
            sigma112: (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mm2))**2,
            sigma113: (np.sqrt(mNu2) + np.sqrt(mt2) + np.sqrt(mm2))**2,
            sigma114: (np.sqrt(mNu2) + np.sqrt(me2) + np.sqrt(mt2))**2,
            sigma115: (np.sqrt(mNu2) + np.sqrt(mt2) + np.sqrt(mm2))**2,
            sigma116: (np.sqrt(mNu2) + np.sqrt(4*me2))**2,
            sigma117: (np.sqrt(mNu2) + np.sqrt(4*mm2))**2,
            sigma118: (np.sqrt(mNu2) + np.sqrt(4*mt2))**2,
            sigma119: (np.sqrt(mUp2) + np.sqrt(me2) + np.sqrt(mDown2))**2,
            sigma120: (np.sqrt(mUp2) + np.sqrt(mm2) + np.sqrt(mDown2))**2,
            sigma121: (np.sqrt(mUp2) + np.sqrt(mt2) + np.sqrt(mDown2))**2
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
    
    folder = 'data/NeutrinoInteractions/NeutrinoPhotonTrident/' + name
    if not os.path.exists(folder):
        os.makedirs(folder)

    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = np.logspace(4, 23, 2 ** 18 + 1) * eV**2  
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
    skin = np.logspace(4, 23, 380000 + 1) * eV**2 
    skin = skin[skin > skin_min]

    xs = getTabulatedXS(sigma, skin)
    rate = interactionRate.calc_rate_s(skin, xs, E, field, cdf=True)

    # downsample
    skin_save = np.logspace(4, 23, 190 + 1) * eV**2 
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
    
    for field in reduced_fields:
        print(field.name)
        process(sigma101, field, 'NeutrinoPhotonNeutrinoElectronPair')
        process(sigma102, field, 'NeutrinoPhotonNeutrinoMuonPair')
        process(sigma103, field, 'NeutrinoPhotonNeutrinoTauPair')
        process(sigma104, field, 'NeutrinoPhotonNeutrinoDownPair')
        process(sigma105, field, 'NeutrinoPhotonNeutrinoUpPair')
        process(sigma106, field, 'NeutrinoPhotonNeutrinoStrangePair')
        process(sigma107, field, 'NeutrinoPhotonNeutrinoCharmPair')
        process(sigma108, field, 'NeutrinoPhotonNeutrinoBottomPair')
        process(sigma109, field, 'NeutrinoPhotonNeutrinoTopPair')
        process(sigma110, field, 'NeutrinoElPhotonElMu')
        process(sigma111, field, 'NeutrinoElPhotonElTa')
        process(sigma112, field, 'NeutrinoMuPhotonMuEl')
        process(sigma113, field, 'NeutrinoMuPhotonMuTa')
        process(sigma114, field, 'NeutrinTaPhotonTaEl')
        process(sigma115, field, 'NeutrinoTaPhotonTaMu')
        process(sigma116, field, 'NeutrinoElPhotonElPair')
        process(sigma117, field, 'NeutrinoMuPhotonMuPair')
        process(sigma118, field, 'NeutrinoTaPhotonMTaPair')
        process(sigma119, field, 'NeutrinoElPhotonUpDown')
        process(sigma120, field, 'NeutrinoMuPhotonUpDown')
        process(sigma121, field, 'NeutrinoTaPhotonUpDown')
        
        