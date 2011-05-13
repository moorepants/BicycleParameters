'''Calculates the moment of inertia about the steer axis of the fork, handlebar
and front wheel.'''
import BicycleParameters as bp

rigid = bp.Bicycle('Browser', pathToBicycles='../bicycles', forceRawCalc=True)
rigPar = rigid.parameters['Benchmark']

masses = bp.np.array([rigid.parameters['Benchmark']['mH'],
                      rigid.parameters['Benchmark']['mF']])

coords = bp.np.array([[rigPar['xH'], rigPar['w']],
                      [0., 0.],
                      [rigPar['zH'], -rigPar['rF']]])

mHF, cHF = bp.total_com(coords, masses)

IH = bp.part_inertia_tensor(rigid.parameters['Benchmark'], 'H')
IF = bp.part_inertia_tensor(rigid.parameters['Benchmark'], 'F')

dH = bp.np.array([rigPar['xH'] - cHF[0], 0., rigPar['zH'] - cHF[2]])
dF = bp.np.array([rigPar['w'] - cHF[0], 0., -rigPar['rF'] - cHF[2]])

IHF = (bp.parallel_axis(IH, rigPar['mH'], dH) +
       bp.parallel_axis(IF, rigPar['mF'], dF))

IHFrot = bp.rotate_inertia_tensor(IHF, rigid.parameters['Benchmark']['lam'])

print "Steer assembly moment of inertia = ", IHFrot[2, 2], 'kg*m**2'
