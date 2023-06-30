import numpy as np

eqp_name = 'eqp.dat'
min_band = 230
max_band = 240

f = open(eqp_name,'r')
lines = f.readlines()
f.close()

nbnd = int(lines[0].split()[-1])
nk = int(len(lines)/(nbnd+1))

print('number of bands:', nbnd)
print('number of kpoints:', nk)

data_GW = np.zeros((nk,max_band - min_band + 1))
data_LDA = np.zeros((nk,max_band - min_band + 1))

line = 0
for i in range(nk):
    line += 1
    omit = 0
    for j in range(nbnd):
        if min_band <= int(lines[line].split()[1]) <= max_band:
            data_LDA[i, j-omit] = lines[line].split()[-2]
            data_GW[i, j-omit]   = lines[line].split()[-1]
        else:
            omit += 1
        line += 1

data_GW.sort(axis=1)
data_LDA.sort(axis=1)

f = open('band.dat','w')
for i in range(max_band - min_band + 1):
    for j in range(nk):
        f.write("%s %s %s\n"%(j+1, data_LDA[j,i], data_GW[j,i]))
    f.write('\n')

f.close()