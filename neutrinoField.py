import numpy as np
from crpropa import eV, erg, c_light, h_planck, k_boltzmann, hertz, ccm, centimeter, joule
import os
import gitHelp as gh
import pandas as pd

cdir = os.path.split(__file__)[0]
datadir = os.path.join(cdir, 'tables/')

class NeutrinoField(object):
    """Base class for neutrino fields"""

    def __init__(self):
        self.name = 'NeutrinoField'
        self.info = 'Base class neutrino field'
        self.energy = [] #[eV]
        self.redshift = None
        self.neutrinoDensity = [] #[eV^-1 cm^-3]
        self.particleID = [] # only is is nubar or nu? 
        self.outdir = 'data/Scaling'

    def createFiles(self):
        try:
            git_hash = gh.get_git_revision_hash()
            addHash = True
        except:
            addHash = False

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        with open(self.outdir + "/" + self.name + "_neutrinoEnergy.txt", 'w') as f:
            f.write('# '+self.info+'\n')
            if addHash: f.write("# Produced with crpropa-data version: "+git_hash+"\n")
            f.write("# neutrino energies in [J]\n")
            for e in self.energy:
                f.write("{}\n".format(e * eV))  # [J]
        if self.redshift is not None:
            with open(self.outdir + "/" + self.name + "_redshift.txt", 'w') as f:
                f.write('# '+self.info+'\n')
                if addHash: f.write("# Produced with crpropa-data version: "+git_hash+"\n")
                f.write("# redshift\n")
                for z in self.redshift:
                    f.write("{}\n".format(np.round(z, 2)))
        with open(self.outdir + "/" + self.name + "_neutrinoDensity.txt", 'w') as f:
            f.write('# '+self.info+'\n')
            if addHash: f.write("# Produced with crpropa-data version: "+git_hash+"\n")
            f.write("# Comoving neutrino number density in [m^-3], format: d(e1,z1), ... , d(e1,zm), d(e2,z1), ... , d(e2,zm), ... , d(en,zm)\n")
            for i, densSlice in enumerate(self.neutrinoDensity):
                #Including redshift evolution
                try:
                    for d in densSlice:
                        f.write("{}\n".format(d * self.energy[i] / ccm))  # [# / m^3], comoving
                #When no redshift is included the densSlice is a 1d array
                except TypeError:
                    f.write("{}\n".format(densSlice * self.energy[i] / ccm))  # [# / m^3], comoving
        print("done: " + self.name)
        
class NeutrinoFieldMassive(object):
    """Base class for massive neutrino fields"""

    def __init__(self):
        self.name = 'NeutrinoField'
        self.info = 'Base class for massive neutrino field'
        self.momentum = [] #[eV / c]
        self.mass = [] #[eV / c^2]
        self.redshift = None
        self.neutrinoDensity = [] #[eV^-1 cm^-3 / c] 
        self.particleID = [] # only is is nubar or nu?
        self.outdir = 'data/Scaling'

    def createFiles(self):
        try:
            git_hash = gh.get_git_revision_hash()
            addHash = True
        except:
            addHash = False

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        with open(self.outdir + "/" + self.name + "_massiveNeutrinoMomentum.txt", 'w') as f:
            f.write('# '+self.info+'\n')
            if addHash: f.write("# Produced with crpropa-data version: "+git_hash+"\n")
            f.write('# neutrino mass: '+self.mass+' [eV / c^2]\n')
            f.write("# neutrino momenta in [J / c]\n")
            for p in self.momentum:
                f.write("{}\n".format(p * eV / c_light))  # [J / c]
        if self.redshift is not None:
            with open(self.outdir + "/" + self.name + "_redshift.txt", 'w') as f:
                f.write('# '+self.info+'\n')
                if addHash: f.write("# Produced with crpropa-data version: "+git_hash+"\n")
                f.write("# redshift\n")
                for z in self.redshift:
                    f.write("{}\n".format(np.round(z, 2)))
        with open(self.outdir + "/" + self.name + "_massiveNeutrinoDensity.txt", 'w') as f:
            f.write('# '+self.info+'\n')
            if addHash: f.write("# Produced with crpropa-data version: "+git_hash+"\n")
            f.write("# Comoving neutrino number density in [m^-3], format: d(e1,z1), ... , d(e1,zm), d(e2,z1), ... , d(e2,zm), ... , d(en,zm)\n")
            for i, densSlice in enumerate(self.neutrinoDensity):
                #Including redshift evolution
                try:
                    for d in densSlice:
                        f.write("{}\n".format(d * self.momentum[i] / ccm))  # [# / m^3], comoving
                #When no redshift is included the densSlice is a 1d array
                except TypeError:
                    f.write("{}\n".format(densSlice * self.momentum[i] / ccm))  # [# / m^3], comoving
        print("done: " + self.name)
        
# --------------------------------------------------------
# interfaces
# --------------------------------------------------------

'''
#for explicit calculation of the integral in energy + angle (computationally expensive to do, no factorization!)
class CnuBmassiveEnergy(NeutrinoField): # 
    """
    Cosmic neutrino background
    """
    
    def __init__(self):
        super(CnuB, self).__init__()
        self.name = 'CnuB'
        self.info = 'Cosmic Neutrino Background, T_CnuB = 1.94535456 K'
        self.T_CnuB = 1.94535456  # CnuB temperature [K]
        self.energy = np.logspace(-10, -1, 101) # [eV]
        self.neutrinoDensity = self.getDensity(self.energy * eV) / 2. * (eV * ccm) # [1/eVcm^3], divided by 2 because it is the flux nu+nux!

    def getDensity(self, eps, z=0, mass2=0):
        """
        Comoving spectral number density dn/deps [1/m^3/J] at given neutrino energy eps [J] and redshift z.
        Multiply with (1+z)^3 for the physical number density.
        """
        return 4*np.pi / c_light**3 / h_planck**3 * np.sqrt(eps*eps - mass2) * eps / (np.exp(np.sqrt(eps*eps - mass2) / (k_boltzmann * self.T_CnuB * (1 + z))) + 1) #* (1 + z) * (1 + z) * (1 + z) # it follows Fermi-Dirac distribution!

    def getEmin(self, z=0):
        """Minimum effective neutrino energy in [J]"""
        return 1e-10 * (1 + z) * eV # the  minimum energy has to bes always larger than mass_nu * c_light * c_light

    def getEmax(self, z=0):
        """Maximum effective neutrino energy in [J]"""
        return 10. * (1 + z) * eV
''' 

class CnuB(NeutrinoField):
    """
    Massless cosmic neutrino background
    """
    
    def __init__(self):
        super(CnuB, self).__init__()
        self.name = 'CnuB'
        self.info = 'Cosmic Neutrino Background, T_CnuB = 1.94535456 K'
        self.T_CnuB = 1.94535456  # CnuB temperature [K]
        self.energy = np.logspace(-10, -1, 101) # [eV]
        self.particleID = +1 # only is is nubar or nu? 
        self.neutrinoDensity = self.getDensity(self.energy * eV) / 2. * (eV * ccm) # [1/eVcm^3], divided by 2 because it is the flux nu+nux!

    def getDensity(self, eps, z=0):
        """
        Comoving spectral number density dn/deps [1/m^3/J] at given neutrino energy eps [J] and redshift z.
        Multiply with (1+z)^3 for the physical number density.
        """
        # to take back to the massless case! The same as the photon background!
        return 4*np.pi / c_light**3 / h_planck**3 * eps * eps / (np.exp(eps / (k_boltzmann * self.T_CnuB * (1 + z))) + 1) * (1 + z) ** 3
                # if massless it does not follow Fermi-Dirac distribution?

    def getEmin(self, z=0):
        """Minimum effective neutrino energy in [J]"""
        return 1e-10 * (1 + z) * eV # the  minimum energy has to bes always larger than mass_nu * c_light * c_light

    def getEmax(self, z=0):
        """Maximum effective neutrino energy in [J]"""
        return 10. * (1 + z) * eV

class CnuB_m1(NeutrinoFieldMassive):
    """
    Massive (m1) cosmic neutrino background
    """
    
    def __init__(self):
        super(CnuB_m1, self).__init__()
        self.name = 'CnuBm1'
        self.info = 'massive Cosmic Neutrino Background, T_CnuB = 1.94535456 K'
        self.T_CnuB = 1.94535456   # CnuB temperature [K]
        self.mass = 0 * eV / c_light / c_light # CnuB mass [kg], set to m2 of 1910.11878
        self.momentum = np.logspace(0, 100, 101) # [J / c]
        self.particleID = +1 # only is is nubar or nu? 
        self.neutrinoDensity = self.getDensity(self.momentum * eV) * ccm / centimeter # [cm /(J cm^3 s)], pure momenta
        # divided by 2 because it is the flux nu+nux!

    def getDensity(self, momentum, z=0):
        """
        Comoving spectral number density dn/dp [1/m^3/J] at given neutrino momentum p [J / c] and redshift z.
        (Multiply with (1+z)^3 for the physical number density. Is it embedded in the temperature dependence on the redshift? No, it determines the peak of the distribution.)
        """
        return 4*np.pi / h_planck**3 / c_light * momentum * momentum / (np.exp(momentum * c_light / (k_boltzmann * self.T_CnuB * (1 + z))) + 1) * (1 + z) ** 3 
        
    def getMass(self):
        """Neutrino mass in [J / c^2]"""
        return self.mass 
    
    def getPmin(self, z=0):
        """Minimum neutrino momentum in [J / c]"""
        return 1e-33 * eV / c_light * (1 + z) 

    def getPmax(self, z=0):
        """Maximum neutrino momentum in [J / c]"""
        return 1e1 * eV / c_light * (1 + z) # to check!

class CnuB_m2(NeutrinoFieldMassive):
    """
    Massive (m2) cosmic neutrino background
    """
    
    def __init__(self):
        super(CnuB_m2, self).__init__()
        self.name = 'CnuBm2'
        self.info = 'massive Cosmic Neutrino Background, T_CnuB = 1.94535456 K'
        self.T_CnuB = 1.94535456   # CnuB temperature [K]
        self.mass = 8.6e-3 * eV / c_light / c_light # CnuB mass [kg], set to m2 of 1910.11878
        self.momentum = np.logspace(0, 100, 101) # [J / c]
        self.particleID = +1 # only is is nubar or nu? 
        self.neutrinoDensity = self.getDensity(self.momentum * eV) * ccm / centimeter / 2 # [cm /(J cm^3 s)], pure momenta
        # divided by 2 because it is the flux nu+nux!

    def getDensity(self, momentum, z=0):
        """
        Comoving spectral number density dn/dp [1/m^3/J] at given neutrino momentum p [J / c] and redshift z.
        (Multiply with (1+z)^3 for the physical number density. Is it embedded in the temperature dependence on the redshift?)
        (maybe I need a further * 4 * np.pi, because of the 3D integration.)
        """
        return 4*np.pi / h_planck**3 / c_light * momentum * momentum / (np.exp(momentum * c_light / (k_boltzmann * self.T_CnuB * (1 + z))) + 1) * (1 + z) ** 3 
        
    def getMass(self):
        """Neutrino mass in [J / c^2]"""
        return self.mass 
    
    def getPmin(self, z=0):
        """Minimum neutrino momentum in [J / c]"""
        return 1e-33 * eV / c_light * (1 + z) 

    def getPmax(self, z=0):
        """Maximum neutrino momentum in [J / c]"""
        return 1e1 * eV / c_light * (1 + z) # to check!

class CnuB_m3(NeutrinoFieldMassive):
    """
    Massive (m3) cosmic neutrino background
    """
    
    def __init__(self):
        super(CnuB_m3, self).__init__()
        self.name = 'CnuBm3'
        self.info = 'massive Cosmic Neutrino Background, T_CnuB = 1.94535456 K'
        self.T_CnuB = 1.94535456  # CnuB temperature [K]
        self.mass = 50e-3 * eV / c_light / c_light # CnuB mass [kg], set to m3 of 1910.11878
        self.momentum = np.logspace(0, 100, 101) # [J / c]
        self.particleID = +1 # only if it is nubar or nu? 
        self.neutrinoDensity = self.getDensity(self.momentum * eV) * ccm / centimeter # [cm /(J cm^3 s)], pure momenta
        # divided by 2 because it is the flux nu+nux!

    def getDensity(self, momentum, z=0):
        """
        Comoving spectral number density dn/dp [1/m^3/J] at given neutrino momentum p [J / c] and redshift z.
        (Multiply with (1+z)^3 for the physical number density. Is it embedded in the temperature dependence on the redshift?)
        """
        return 4*np.pi / h_planck**3 / c_light * momentum * momentum / (np.exp(momentum * c_light / (k_boltzmann * self.T_CnuB * (1 + z))) + 1) * (1 + z) ** 3
        
    def getMass(self):
        """Neutrino mass in [J / c^2]"""
        return self.mass 
    
    def getPmin(self, z=0):
        """Minimum neutrino momentum in [J / c]"""
        return 1e-33 * eV / c_light * (1 + z) 

    def getPmax(self, z=0):
        """Maximum neutrino momentum in [J / c]"""
        return 1e1 * eV * (1 + z) / c_light 

'''    
class CnuxB_m3(NeutrinoFieldMassive):
    """
    Massive (m3) cosmic neutrino background
    """
    
    def __init__(self):
        super(CnuB, self).__init__()
        self.name = 'massiveCnuB'
        self.info = 'massive Cosmic (Anti)neutrino Background, T_CnuB = 1.94535456 K'
        self.T_CnuB = 1.94535456  # CnuB temperature [K]
        self.mass = 50e-3 * eV / c_light / c_light # CnuB mass [kg], set to m3 of 1910.11878
        self.momentum = np.logspace(0, 100, 101) # [J / c]
        self.particleID = -1 # only is is nubar or nu? 
        self.neutrinoDensity = self.getDensity(self.momentum * eV) * ccm / centimeter / 2 # [cm /(J cm^3 s)], pure momenta
        # divided by 2 because it is the flux nu+nux!

    def getDensity(self, momentum, z=0):
        """
        Comoving spectral number density dn/dp [1/m^3/J*m/s] at given neutrino momentum p [J / c] and redshift z.
        (Multiply with (1+z)^3 for the physical number density. Is it embedded in the temperature dependence on the redshift?)
        """
        return 4*np.pi / h_planck**3 * momentum * momentum / (np.exp(momentum * c_light / (k_boltzmann * self.T_CnuB * (1 + z))) + 1) 
        
    def getMass(self, z=0):
        """Neutrino mass in [J / c^2]"""
        return self.mass 
    
    def getPmin(self, z=0):
        """Minimum neutrino momentum in [J / c]"""
        return 0

    def getPmax(self, z=0):
        """Maximum neutrino momentum in [J / c]"""
        return 100 * (1 + z) * eV / c_light
'''





