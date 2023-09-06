"""Constants"""
# file stolen from Electron-phononcoupling

# Tolerance criterions
tol5 = 1E-5
tol6 = 1E-6
tol8 = 1E-8
tol12 = 1E-12

# Conversion factor
Ha2eV = 27.2113961318
Ry2eV = 13.6056980659
Ry2meV = Ry2eV*1000
eV2Ry = 1./Ry2eV
Ry2cm1 = Ry2eV * 8065.541
Ry2Thz = 3.289828E3
a2bohr = 1.88973

# Boltzman constant
kb_HaK = 3.1668154267112283e-06

# Electron mass over atomical mass unit
me_amu = 5.4857990965007152E-4

# We use Ry in energy so time unit = hbar/Ry
# convert time unit to nanosec
# hbar/hartree = 2.418884326505*10^{-17} sec
t2nsec = 2.418884326505*2*1E-8
t2fs = 2.418884326505*2*1E-2

mev2ps = 0.6582119514 # 1000/ ((1/hbar)*1e-12)
