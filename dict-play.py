import bicycleparameters as bp
import os
from collections import OrderedDict

def new_par(bike_name):
    bike = bp.Bicycle(bike_name, pathToData=os.getcwd()+'\\data')
    par = bike.parameters['Benchmark']
    parPure = bp.io.remove_uncertainties(par)
    return parPure

WHEEL_COLUMNS=[{'name': 'Front Wheel', 'id': 'fW'},
               {'name': 'Rear Wheel', 'id': 'rW'}]

WHEEL_LABELS=['Radius',
              'Mass',
              'Moment Ixx',
              'Moment Iyy']

FRAME_COLUMNS=[{'name': '', 'id': 'label'},
               {'name': 'Rear Body', 'id': 'rB'},
               {'name': 'Front Assembly', 'id': 'fA'}]    

FRAME_LABELS=['Center-of-Mass X',
              'Center-of-Mass Y',
              'Total Mass',
              'Moment Ixx',
              'Moment Iyy',
              'Moment Izz',
              'Moment Ixz']

GENERAL_COLUMNS=[{'name': '', 'id': 'label'},
                 {'name': 'Value', 'id': 'val'}]

GENERAL_LABELS=['Wheel Base', 
                'Trail',
                'Steer Axis Tilt',
                'Gravity']
pList=['rF', 'mF', 'IFxx', 'IFyy', 'rR', 'mR', 'IRxx', 'IRyy',
       'w', 'c', 'lam', 'g',
       'xB', 'zB', 'mB', 'IBxx', 'IByy', 'IBzz', 'IBxz', 'xH', 'zH', 'mH', 'IHxx', 'IHyy', 'IHzz', 'IHxz',]

'''
def genDataDic():
    data_dic = OrderedDict()
    parPure = new_par('Benchmark')
    index = 0
    for i in WHEEL_COLUMNS:     
        for p in range(index,8):   
            data_dic[pList[p]] = {i['id']:parPure.get(pList[p])}
            index += 1
            if index == 4:
                break
    return data_dic

dic = genDataDic()
print(dic)
'''
def wheel_data():
    parPure = new_par('Benchmark')
    data = []
    empty = []
    fW = []
    rW = []
    for i in WHEEL_LABELS:
        empty.append({'label': i})
    for i in range(8):
        if i < 4:
            fW.append({'fW':parPure.get(pList[i])}) 
        else:
            rW.append({'rW':parPure.get(pList[i])})
    for c, d, e in zip(empty, fW, rW):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        zipped.update(e)
        data.append(zipped)
    return data

def frame_data():
    parPure = new_par('Benchmark')
    data = []
    empty = []
    rB = []
    fA = []
    for i in FRAME_LABELS:
        empty.append({'label': i})
    for i in range(12, len(pList)):
        if i < 19:
            rB.append({'rB':parPure.get(pList[i])}) 
        else:
            fA.append({'fA':parPure.get(pList[i])})
    for c, d, e in zip(empty, rB, fA):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        zipped.update(e)
        data.append(zipped)
    return data

def general_data():
    parPure = new_par('Benchmark')
    data = []
    labels = []
    val = []
    for i in GENERAL_LABELS:
        labels.append({'label': i})
    for i in range(8, 12):
        val.append({'val':parPure.get(pList[i])}) 
    for c, d in zip(labels, val):
        zipped = {}
        zipped.update(c)
        zipped.update(d)
        data.append(zipped)
    return data

def compile_data():
    return wheel_data()+frame_data()+general_data()

def draw():
    wheelData = wheel_data()
    wheelData[0]['fW'] = .2
    frameData = frame_data()
    genData = general_data()
    newP = []
    currentBike = bp.Bicycle('Benchmark', pathToData=os.getcwd()+'\\data')
    for p in range(8):
        if p < 4:
            newP.extend([pList[p], wheelData[p].get('fW')]) 
        else:
            newP.extend([pList[p], wheelData[p-4].get('rW')])
    for p in range(12, len(pList)):
        if p < 19:
            newP.extend([pList[p], frameData[p-12].get('rB')])
        else: 
            newP.extend([pList[p], frameData[p-19].get('fA')])
    for p in range(8,12):
        newP.extend([pList[p], genData[p-8].get('val')])
    for i in range(0,len(newP),2):
        currentBike.parameters['Benchmark'][newP[i]] = newP[i+1]
    return currentBike.plot_bicycle_geometry(show=True)

draw()
input('type')
