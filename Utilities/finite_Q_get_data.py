import os


nQ = 1
number_kpt = 144

os.chdir('5_Q')
os.system("echo 'eigenvalues' > xc_%s.dat "%nQ)
os.system("echo 'kx ky kz' > kpt_crystal.dat")
for i in range(2,number_kpt + 1):
    print(i)
    os.system("sed -n %sp Q-%s/5.2-absorp-Q/eigenvalues.dat|awk '{print $1}' >> xc_%s.dat"%(nQ+4, i, nQ))
    os.system("cat Q-%s/kpt >> kpt_crystal.dat"%i)
