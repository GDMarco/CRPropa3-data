import numpy as np
from crpropa import eV, erg, c_light, h_planck, k_boltzmann, hertz, ccm
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
        self.particleID = [] 
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
        
# --------------------------------------------------------
# interfaces
# --------------------------------------------------------
class CnuB(NeutrinoField):
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

    def getDensity(self, eps, z=0):
        """
        Comoving spectral number density dn/deps [1/m^3/J] at given neutrino energy eps [J] and redshift z.
        Multiply with (1+z)^3 for the physical number density.
        """
        return 8*np.pi / c_light**3 / h_planck**3 * eps**2 / np.expm1(eps / (k_boltzmann * self.T_CnuB)) 

    def getEmin(self, z=0):
        """Minimum effective neutrino energy in [J]"""
        return 1e-10 * eV

    def getEmax(self, z=0):
        """Maximum effective neutrino energy in [J]"""
        return 0.1 * eV
