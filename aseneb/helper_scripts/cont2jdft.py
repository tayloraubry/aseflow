#!/usr/bin/env python

#script to generate jdft lattice and ionpos files from poscar
import os,sys
import numpy as np

def readfile(filename):
    #read a file into a list of strings
    f = open(filename,'r')
    tempfile = f.readlines()
    f.close()
    return tempfile

def writefile(filename,tempfile):
    #write tempfile (list of strings) to filename
    f = open(filename,'w')
    f.writelines(tempfile)
    f.close()

def check_file(filename):
    #check if file exists and die if doesn't
    if not os.path.isfile(filename):
        print('\'' + filename + '\' file doesn\'t exist!')
        sys.exit()

def has_sd(tempfile):
    #determine if POSCAR has selective dynamics
    line = 7
    if 's' in tempfile[line].lower()[0] or 'selective' in tempfile[line].lower():
        use_sd = True
    else:
        use_sd = False
    return use_sd

def main():
    
    check_file('CONTCAR')
    tempfile = readfile('CONTCAR')
    
    lattfile = ['lattice']
    latt = tempfile[2:5]
    latt = [x.split() for x in latt]
    latt = [[float(y)*1.88973 for y in x] for x in latt]
    latt = np.array(latt).T.tolist()
    latt = [['{:20.16f}'.format(y) for y in x] for x in latt]
    latt = [' '.join(x) for x in latt]
    [lattfile.append('   ' + x) for x in latt]
    for i in range(0,3):
        lattfile[i] += ' \\ \n'
    lattfile[-1] += '\n'
    writefile('in.lattice',lattfile)

    #get atoms
    line = 5
    ele = tempfile[line].split()

    #get number of atoms
    line = 6
    elenums = tempfile[line].split()
    elenums = [int(float(x)) for x in elenums]
    total = sum(elenums)

    #make list of elements
    elelist = []
    for i in range(len(elenums)):
        for j in range(elenums[i]):
            elelist.append(ele[i])

    use_sd = has_sd(tempfile)
    if use_sd == True:
        line = 9
        print('Ignoring selective dynamics, all atoms will be frozen')
    else:
        line = 8
    ctype = tempfile[line-1].split()[0]
    if ctype[0].lower() == 'c':
        factor = 1.88973
    else:
        factor = 1.0
    coords = []
    sd = []
    for i in range(line,line+total):
        c = tempfile[i].split()[0:3]
        c = [float(x)*factor for x in c]
        c = [str(x) for x in c]
        coords.append(c)
        if use_sd == True:
            sd.append(tempfile[i].split()[3:6])

    ionposfile = []
    for i,c in enumerate(coords):
        E = elelist[i]
        ionline = 'ion ' + E + ' ' + ' '.join(c)
        sdflag = '0'

        ionline += ' ' + sdflag + '\n'
        ionposfile.append(ionline)

    writefile('in.ionpos',ionposfile)

if __name__ == '__main__':
    main()

