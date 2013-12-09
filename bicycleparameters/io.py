#!/usr/bin/env python

import re
import os
import numpy as np
from uncertainties import ufloat
from scipy.io import loadmat

def filename_to_dict(filename):
    '''Returns a dictionay of values based on the pendulum data file name.

    '''
    o = space_out_camel_case(os.path.splitext(filename)[0], output='list')
    # this only accounts for single digit trial numbers
    trial = o[-1][-1]
    o[-1] = o[-1][:-1]
    o.append(trial)
    breakdown = ['bicycle', 'part', 'pendulum', 'angleOrder', 'trial']
    dat = {}
    for word, val  in zip(breakdown, o):
        dat[word] = val
    return dat

def load_parameter_text_file(pathToFile):
    """
    Returns a dictionary of float and/or ufloat parameters from a parameter file.

    Parameters
    ----------
    pathToFile : string
        The path to the text file with the parameters listed in the specified
        format.

    Returns
    -------
    parameters : dictionary
        A dictionary of the values stored in the text files.

    For example::

        c = 0.08 +/- 0.01
        d=0.314+/-0.002
        t = 0.1+/-0.01, 0.12+/-0.02
        whb = 0.5

    The first item on the line must be the variable name and the second is an
    equals sign. The values to the right of the equal sign much may or may not
    contain an uncertainty designated by `+/-`. Multiple comma seperated values
    will be averaged.

    """

    parameters = {}
    # parse the text file
    with open(pathToFile, 'r') as f:
        for line in f:
            # ignore lines that start with a hash
            if line[0] != '#':
                # remove any whitespace characters and comments at the end of
                # the line, then split the right and left side of the equality
                equality = line.strip().split('#')[0].split('=')
                # ['a ', ' 0.1 +/- 0.05 , 0.09 +/- 0.05']
                valList = equality[1].strip().split(',')
                # ['0.1 +/- 0.05 ', ' 0.09 +/- 0.05']
                if '+/-' in equality[1]:
                    values = [ufloat(x) for x in valList]
                else:
                    values = [float(x) for x in valList]
                # store in dictionary
                parameters[equality[0].strip()] = np.mean(values)

    return parameters

def load_pendulum_mat_file(pathToFile):
    '''Returns a dictionay containing the data from the pendulum data mat file.

    '''
    pendDat = {}
    loadmat(pathToFile, mdict=pendDat)
    #clean up the matlab imports
    del(pendDat['__globals__'], pendDat['__header__'], pendDat['__version__'])
    for k, v in pendDat.items():
        try:
            #change to an ascii string
            pendDat[k] = v[0].encode('ascii')
        except:
            #if an array of a single number
            if np.shape(v)[0] == 1:
                pendDat[k] = v[0][0]
            #else if the notes are empty
            elif np.shape(v)[0] == 0:
                pendDat[k] = ''
            #else it is the data which needs to be a one dimensional array
            else:
                pendDat[k] = v.reshape((len(v),))
    return pendDat

def remove_uncertainties(dictionary):
    '''Returns a dictionary with the uncertainties removed.'''
    noUncert = {}
    for k, v in dictionary.items():
        try:
            # this is the case if the value is a single uncertainty
            noUncert[k] = v.nominal_value
        except AttributeError:
            # this is the case if the value is an array of ufloats
            try:
                noUncert[k] = [x.nominal_value for x in v]
            except TypeError:
                # this is the case if the value is a float
                noUncert[k] = v
    return noUncert

def write_parameter_text_file(pathToTxtFile, parDict):
    '''Writes parameter set to file.

    Parameters
    ----------
    pathToTxtFile : string
        The path to the file to write the parameters.
    pardict : dictionary
        A dictionary of parameters for the bicycle.

    Returns
    -------
    saved : boolean
        True if the file was saved and false if not.

    '''

    # make the Parameters directory if it doesn't exist
    head, tail = os.path.split(pathToTxtFile)
    if not os.path.isdir(head):
        print "Created direcotry %s" % head
        os.makedirs(head)

    try:
        f = open(pathToTxtFile)
        f.close()
        del f
        ans = None
        while ans !=  'y' and ans != 'n':
            ans = raw_input("%s exists already. Are you sure you want" \
                            " to overwrite it? (y or n)\n" % pathToTxtFile)
        if ans == 'y':
            f = open(pathToTxtFile, 'w')
    except IOError:
        f = open(pathToTxtFile, 'w')

    try:
        f
        keys = sorted(parDict.keys())
        for key in keys:
            f.write(key + ' = ' + str(parDict[key]) + '\n')
        f.close()
        print "Parameters saved to %s" % pathToTxtFile
        return True
    except UnboundLocalError:
        print "%s was not saved." % pathToTxtFile
        return False

def space_out_camel_case(s, output='string'):
        """Adds spaces to a camel case string.  Failure to space out string
        returns the original string.

        Examples
        --------
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC')
        'DMLS Services Other BS Text LLC'
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC', output='list')
        ['DMLS', 'Services', 'Other', 'BS', 'Text', 'LLC']

        """
        if output == 'string':
            return re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ',
                          s).strip()
        elif output == 'list':
            string = re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ',
                            s).strip()
            return string.split(' ')
        else:
            raise ValueError

def write_periods_to_file(pathToRawFile, mp):
    '''Writes the provided periods to file.

    Parameters
    ----------
    pathToRawFile : string
        The path to the <bicycle name>Measured.txt file
    mp : dictionary
        The measured parameters dictionary. Should contain complete period
        data.

    '''

    # clear any period data from the file
    f = open(pathToRawFile, 'r')
    baseData = ''
    for line in f:
        if not line.startswith('T'):
            baseData += line
    f.close()
    # add the periods to the base data
    periodKeys = [x for x in mp.keys() if x.startswith('T')]
    periodKeys.sort()
    withPeriods = baseData
    for k in periodKeys:
        withPeriods += k + ' = ' + str(mp[k]) + '\n'

    # write it to the file
    f = open(pathToRawFile, 'w')
    f.write(withPeriods)
    f.close()
