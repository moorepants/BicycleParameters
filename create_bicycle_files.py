import pickle
import os
from uncertainties import unumpy
from numpy import ones

f = open('bicycles/udata.p', 'r')
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
f = []
# open the files for writing
for i, bike in enumerate(shortnames):
    if not os.path.isdir('bicycles/' + bike):
        os.system('mkdir bicycles/' + bike)
    f.append(open('bicycles/' + bike + '/' + bike +
            'Measured.txt', 'w'))
    f[i].write('shortname=' + bike + '\n')

for k, v in udata.items():
    try:
        if v.shape[0] == 3:
            # v is a (3, 8) array
            vn = v.T
        else:
            # v is a (8,) array
            vn = v
    except AttributeError:
        # v isn't an array
        vn = v
    if k == 'lRod' or k == 'mRod' or k == 'rRod':
        # these are not arrays, so make them one
        vn = ones(8, dtype='object')*v
    for i, val in enumerate(vn):
        if k == 'bikes':
            k = 'name'
        try:
            nom = val.nominal_value
            std = val.std_dev()
            line = k + '=' + str(nom) + ',' + str(std) + '\n'
        except AttributeError:
            try:
                nom = unumpy.nominal_values(val)
                std = unumpy.std_devs(val)
                line = k + '=[' + ','.join(str(nom)[1:-1].split()) + '],[' + ','.join(str(std)[1:-1].split()) + ']\n'
            except:
                line = k + '=' + str(val) + '\n'
        print shortnames[i], line
        f[i].write(line)

lines = ('FrameTorsionalT=\n' +
        'FrameCompoundT=\n' +
        'ForkTorsionalT=\n' +
        'ForkCompoundT=\n' +
        'RwheelTorsionalT=\n' +
        'RwheelCompoundT=\n' +
        'FwheelTorsionalT=\n' +
        'FwheelCompoundT=\n')

for openfile in f:
    openfile.write(lines)
    openfile.close()
