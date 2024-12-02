"""units.py

Python module to implement physical constants
Definition of SI base units and constants used elsewhere in the code
 Based on:
 - CODATA recommended values of the fundamental physical constants: 2006
 	doi:10.1103/RevModPhys.80.633
 - IAU 2012 Resolution B2, IAU 2015 Resolution B2
 	https:#www.iau.org/administration/resolutions/

Note:
This file is copied from CRPropa3's Units.h file. It should be synched 
with the file that can be found here: 
https:#github.com/CRPropa/CRPropa3/blob/master/include/crpropa/Units.h
"""

from numpy import pi

# SI units
meter = 1
second = 1
kilogram = 1
ampere = 1
mol = 1
kelvin = 1

# derived units
newton = 1 * kilogram * meter / second / second
pascal = 1 * newton / meter / meter
joule = 1 * newton * meter
tesla = 1 * newton / ampere / meter
volt = 1 * kilogram * meter * meter / ampere / second / second / second
coulomb = 1 * ampere * second
hertz = 1 / second
rad = 1
deg = pi / 180.

# SI Prefixes
yocto = 1E-24
zepto = 1E-21
atto = 1E-18
femto = 1E-15
pico = 1E-12
nano = 1E-9
micro = 1E-6
milli = 1E-3

kilo = 1E3
mega = 1E6
giga = 1E9
tera = 1E12
peta = 1E15
exa = 1E18
zetta = 1E21
yotta = 1E24


# physical constants
eplus = 1.602176487e-19 * ampere * second
c_light = 2.99792458e8 * meter / second
c_squared = c_light * c_light
amu = 1.660538921e-27 * kilogram
mass_proton = 1.67262158e-27 * kilogram
mass_neutron = 1.67492735e-27 * kilogram
mass_electron = 9.10938291e-31 * kilogram
mass_muon = 1.883531627e-28 * kilogram
mass_tauon = 3.167e-27 * kilogram
h_planck = 6.62606957e-34 * joule * second
k_boltzmann = 1.3806488e-23 * joule / kelvin
mu0 = 4 * pi * 1e-7 * newton / ampere / ampere
epsilon0 = 1.0 / mu0 / c_squared * ampere * second / volt / meter
alpha_finestructure = eplus * eplus / 2. / epsilon0 / h_planck / c_light

radius_electron = eplus * eplus / 4. / pi / epsilon0 / mass_electron / c_squared
sigma_thomson = 8. * pi / 3. * radius_electron * radius_electron

radius_muon = eplus * eplus / 4. / pi / epsilon0 / mass_muon / c_squared
sigma_thomson_muon = 8. * pi / 3. * radius_muon * radius_muon

radius_tauon = eplus * eplus / 4. / pi / epsilon0 / mass_tauon / c_squared
sigma_thomson_tauon = 8. * pi / 3. * radius_tauon * radius_tauon


#mass W boson
massWGeV = 80.379
mWkg = 1.43288582 * 1e-25 #kg
mW2 = (mWkg*c_light**2.) ** 2 #squared W mass [J^2/c^4]

#mass charged pion
massPionGeV = 139.57039e-3 #GeV/c^2
mass_charged_pion = mWkg * massPionGeV / massWGeV #kg

#mass charged kaon 
massKaonGeV = 493.677e-3 #GeV/c^2
mass_charged_kaon = mWkg * massKaonGeV / massWGeV #kg

# mass neutral pion 
massNPionGeV = 134.9768e-3 #GeV/c^2
mass_neutral_pion = mWkg * massNPionGeV / massWGeV #kg

# decay width pi0
tauNP = 8.43e-17 * second
widthNP = h_planck / 2 / pi / tauNP

# mass neutral kaon 
massNKaonGeV = 497.611e-3 #GeV/c^2
mass_neutral_kaon = mWkg * massNKaonGeV / massWGeV #kg

massQuarkUpGeV = 2.3e-3 #GeV/c^2
massUpkg = mWkg * massQuarkUpGeV / massWGeV #kg
mUp2 = (massUpkg*c_light**2.) ** 2 # squared up quark mass [J^2/c^4]

massQuarkDownGeV = 4.8e-3 #GeV/c^2
massDownkg = mWkg * massQuarkDownGeV / massWGeV #kg
mDown2 = (massDownkg*c_light**2.) ** 2 # squared up quark mass [J^2/c^4]

massQuarkCharmGeV = 1275e-3 #GeV/c^2
massCharmkg = mWkg * massQuarkCharmGeV / massWGeV #kg
mCharm2 = (massCharmkg*c_light**2.) ** 2 # squared up quark mass [J^2/c^4]

massQuarkStrangeGeV = 95e-3 #GeV/c^2
massStrangekg = mWkg * massQuarkStrangeGeV / massWGeV #kg
mStrange2 = (massStrangekg*c_light**2.) ** 2 # squared up quark mass [J^2/c^4]

massQuarkTopGeV = 173210e-3 #GeV/c^2
massTopkg = mWkg * massQuarkTopGeV / massWGeV #kg
mTop2 = (massTopkg*c_light**2.) ** 2 # squared up quark mass [J^2/c^4]

massQuarkBottomGeV = 4180e-3 #GeV/c^2
massBottomkg = mWkg * massQuarkBottomGeV / massWGeV #kg
mBottom2 = (massBottomkg*c_light**2.) ** 2 # squared up quark mass [J^2/c^4]

# gauss
gauss = 1e-4 * tesla
microgauss = 1e-6 * gauss
nanogauss = 1e-9 * gauss
muG = microgauss
nG = nanogauss

erg = 1E-7 * joule

# electron volt
electronvolt = eplus * volt
kiloelectronvolt = 1e3 * electronvolt
megaelectronvolt = 1e6 * electronvolt
gigaelectronvolt = 1e9 * electronvolt
teraelectronvolt = 1e12 * electronvolt
petaelectronvolt = 1e15 * electronvolt
exaelectronvolt = 1e18 * electronvolt
eV = electronvolt
keV = kiloelectronvolt
MeV = megaelectronvolt
GeV = gigaelectronvolt
TeV = teraelectronvolt
PeV = petaelectronvolt
EeV = exaelectronvolt

barn = 1E-28 * meter * meter

# astronomical distances
au = 149597870700 * meter
ly = 365.25 * 24 * 3600 * second * c_light
parsec = 648000 / pi * au
kiloparsec = 1e3 * parsec
megaparsec = 1e6 * parsec
gigaparsec = 1e9 * parsec
pc = parsec
kpc = kiloparsec
Mpc = megaparsec
Gpc = gigaparsec

# meter
kilometer = 1000 * meter
centimeter = 0.01 * meter
km = kilometer
cm = centimeter

# second
nanosecond = 1e-9 * second
microsecond = 1e-6 * second
millisecond = 1e-3 * second
minute = 60 * second
hour = 3600 * second
ns = nanosecond
mus = microsecond
ms = millisecond
sec = second

# volume
ccm = cm*cm*cm
