import os
import tarfile
import zipfile

def package_data(pathToData):
    """Packages all the data for BicycleParameters for distribution."""

    tar = tarfile.open('bicycleparameters-data.tar.bz2', 'w:bz2')
    zipFile = zipfile.ZipFile('bicycleparameters-data.zip', 'w')

    # get the bicycle data
    bicycles = ['Benchmark', 'Browser', 'Browserins', 'Crescendo', 'Fisher',
                'Gyro', 'Pista', 'Rigid', 'Silver', 'Stratos', 'Yellow',
                'Yellowrev']
    pathToBicycles = os.path.join(pathToData, 'bicycles')
    for bike in bicycles:
        folders = ['RawData', 'Parameters', 'Photos']
        def raw_file_check(fName, bike):
            return fName.endswith('.mat') or fName.endswith('Measured.txt')
        def par_file_check(fName, bike):
            return fName.startswith(bike) and fName.endswith('.txt')
        def photo_file_check(fName, bike):
            return fName.startswith(bike) and fName.endswith('.jpg')
        rules = [raw_file_check, par_file_check, photo_file_check]
        for folder, rule in zip(folders, rules):
            pathToFolder = os.path.join(pathToBicycles, bike, folder)
            try:
                filesInFolder = os.listdir(pathToFolder)
                print "Searching", pathToFolder
            except OSError:
                print "Did not find:", pathToFolder
            else:
                for f in filesInFolder:
                    if rule(f, bike):
                        pathToFile = os.path.join(pathToFolder, f)
                        print "Added:", pathToFile
                        tar.add(pathToFile,
                                arcname=os.path.join('data', 'bicycles',
                                    pathToFile.split('bicycles')[1][1:]))
                        zipFile.write(pathToFile,
                                  os.path.join('data', 'bicycles',
                                  pathToFile.split('bicycles')[1][1:]))
    # get the rider data
    riders = ['Aurelia', 'Chris', 'Jason', 'Luke', 'Mont']
    pathToRiders = os.path.join(pathToData, 'riders')
    for rider in riders:
        folders = ['Parameters', 'RawData']
        def par_file_check(fName, rider):
            return fName.startswith(rider) and fName.endswith('.txt')
        def raw_file_check(fName, rider):
            return fName.startswith(rider) and (fName.endswith('YeadonCFG.txt') or
                fName.endswith('YeadonMeas.txt'))
        rules = [par_file_check, raw_file_check]
        for folder, rule in zip(folders, rules):
            pathToFolder = os.path.join(pathToRiders, rider, folder)
            try:
                filesInFolder = os.listdir(pathToFolder)
                print "Searching", pathToFolder
            except OSError:
                print "Did not find:", pathToFolder
            else:
                for f in filesInFolder:
                    if rule(f, rider):
                        pathToFile = os.path.join(pathToFolder, f)
                        print "Added:", pathToFile
                        tar.add(pathToFile, arcname=os.path.join('data', 'riders',
                            pathToFile.split('riders')[1][1:]))
                        zipFile.write(pathToFile,
                                  os.path.join('data', 'riders',
                                  pathToFile.split('riders')[1][1:]))


    tar.close()
    zipFile.close()
