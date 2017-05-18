'''Creates indivdual *Measured.txt files for all of the bicycles measured at TU
Delft for use in the new program.'''
import pickle
import os
from uncertainties import unumpy
from numpy import ones

f = open('../../data/udata.p', 'r')
udata = pickle.load(f)
f.close()

shortnames = ['Browser',
              'Browserins',
              'Crescendo',
              'Fisher',
              'Pista',
              'Stratos',
              'Yellow',
              'Yellowrev']

measuredFiles = []
bicyclesFolder = os.path.join('..', 'bicycles')
# open the files for writing
for i, bike in enumerate(shortnames):
    # create the directory if it doesn't exist
    bikeFolder = os.path.join(bicyclesFolder, bike)
    if not os.path.isdir(bikeFolder):
        os.system('mkdir ' + bikeFolder)
    pathToMeasuredFile = os.path.join(bikeFolder, 'RawData',
                                      bike + 'Measured.txt')
    # store the file handle
    measuredFiles.append(open(pathToMeasuredFile, 'w'))

badKeys = ['frontWheelPressure',
           'names',
           'shortnames',
           'totalMass',
           'rearWheelPressure']

# go through the dictionary of raw data and write it to the individual files
for k, v in udata.items():
    print k, v
    isPeriod = k.startswith('T')
    isBad = k in badKeys
    # don't add the periods
    if (not isPeriod) and (not isBad):
        try:
            if v.shape[0] == 3:
                # v is a (3, 8) array
                vn = v.T
                print k, "is a 3x8 array"
            else:
                # v is a (8,) array
                vn = v
                print k, "is a 8x array"
        except AttributeError:
            # v isn't an array
            vn = v
            print k, "is not an array"
        if k == 'lC' or k == 'mC' or k == 'dC':
            # these are not arrays, so make them one
            vn = ones(8, dtype='object') * v
            print "Made", k, "into and array"
            # rename k
            k = k[0] + 'P'
        for i, val in enumerate(vn):
            if k == 'bikes':
                k = 'name'
            try:
                nom = val.nominal_value
                std = val.std_dev()
                line = k + ' = ' + str(nom) + '+/-' + str(std) + '\n'
            except AttributeError:
                try:
                    nom = unumpy.nominal_values(val)
                    line = ''
                    for j, number in enumerate(nom):
                        line += k + str(j + 1) + ' = ' + str(val[j]) + '\n'
                except ValueError:
                    line = k + ' = ' + str(val) + '\n'
            #print shortnames[i], line
            measuredFiles[i].write(line)
    else:
        print "Did not add:", k
    print '\n'

for openfile in measuredFiles:
    openfile.write('g = 9.81 +/- 0.01\n')
    openfile.close()
