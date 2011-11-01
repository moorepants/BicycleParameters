def to_latex(var):
    latexMap = {'lF': 'l_F',
                'lP': 'l_P',
                'nF': 'n_F',
                'lR': 'l_R',
                'mF': 'm_F',
                'mP': 'm_P',
                'mB': 'm_H',
                'mH': 'm_H',
                'mR': 'm_R',
                'mS': 'm_S',
                'mG': 'm_G',
                'f': 'f',
                'dF': 'd_F',
                'dP': 'd_P',
                'w': 'w',
                'dR': 'dR',
                'gamma': '\gamma',
                'g': 'g',
                'lcs': 'l_cs',
                'hbb': 'h_{bb}',
                'lsp': 'l_{sp}',
                'lst': 'l_{st}',
                'lamst': '\lambda_{st}',
                'whb': 'w_{hb}',
                'LhbF': 'l_{hbF}',
                'LhbR': 'l_{hbR}',
                'h1': 'h_1',
                'h2': 'h_2',
                'h3': 'h_3',
                'h4': 'h_4',
                'h5': 'h_5',
                'h6': 'h_6',
                'h7': 'h_7',
                'd1': 'd_1',
                'd2': 'd_2',
                'd3': 'd_3',
                'd4': 'd_4',
                'd5': 'd_5',
                'd6': 'd_6',
                'd': 'd',
                'l': 'l',
                'nR':'n_R'}
    try:
        latex = latexMap[var]
    except KeyError:
        if var.startswith('alpha'):
            latex = r'\alpha_{' + var[-2:] + '}'
        elif var.startswith('a') and len(var) == 3:
            latex = r'\a_{' + var[-2:] + '}'
        elif var.startswith('T'):
            latex = 'T^' + var[1] + '_{' + var[-2:] + '}'
        else:
            raise

    return latex

def generate_bicycle_raw_tables(*bicycles):
    """Generates a table of values for the bicycles."""

    # generate a complete list of the variables
    allVariables = []
    for bicycle in bicycles:
        allVariables += bicycle.parameters['Measured'].keys()
    # remove duplicates and sort
    allVariables = sorted(list(set(allVariables)))

    table = []
    #table[-1] = [' '] + [bicycle.bicycleName for bicycle in bicycles]
    largest = [0]
    for bicycle in bicycles:
        l = len(bicycle.bicycleName) / 2
        largest += [l, l]
    for var in allVariables:
        # add a new line
        table.append([])
        table[-1].append(':math:`' + to_latex(var) + '`')
        for bicycle in bicycles:
            try:
                val, sig = uround(bicycle.parameters['Measured'][var]).split('+/-')
            except ValueError:
                val = str(bicycle.parameters['Measured'][var])
                sig = 'NA'
            except KeyError:
                val = 'NA'
                sig = 'NA'
            table[-1] += [val, sig]
        colSize = [len(string) for string in table[-1]]
        for i, pair in enumerate(zip(colSize, largest)):
            if pair[0] > pair[1]:
                largest[i] = pair[0]

    rstTable = '+' + '-' * (largest[0] + 2)

    for i, bicycle in enumerate(bicycles):
        rstTable += '+' + '-' * (largest[i + 1] + largest[i + 2] + 5)
    rstTable += '+\n|' + ' ' * (largest[0] + 2)

    for i, bicycle in enumerate(bicycles):
        rstTable += '| ' + bicycle.bicycleName + ' ' * (largest[i + 1] +
                largest[i + 2] + 4 - len(bicycle.bicycleName))
    rstTable += '|\n'

    for j, row in enumerate(table[1:]):
        if j == 0:
            dash = '='
        else:
            dash = '-'
        line = ''
        for i in range(len(row)):
            line += '+' + dash * (largest[i] + 2)
        line += '+\n|'
        for i, item in enumerate(row):
            line += ' ' + item + ' ' * (largest[i] - len(item)) + ' |'
        line += '\n'
        rstTable += line

    for num in largest:
       rstTable += '+' + dash * (num + 2)
    rstTable += '+'

    return rstTable


def uround(value):
    '''Round values according to their uncertainity

    Parameters:
    -----------
    value: float with uncertainty

    Returns:
    --------
    s: string that is properly rounded

    2.4563752289999+/-0.0003797273827

    becomes

    2.4564+/-0.0004

    This doesn't work for weird cases like large uncertainties.
    '''
    try:
        # grab the nominal value and the uncertainty
        nom = value.nominal_value
        uncert = value.std_dev()
        # convert the uncertainty to a string
        s = str(uncert)
        # find the first non-zero character
        for j, number in enumerate(s):
            if number == '0' or number == '.':
                pass
            else:
                digit = j
                break
        newUncert = round(uncert, digit-1)
        newNom = round(nom, len(str(newUncert)) - 2)
        newValue = ufloat((newNom, newUncert))
        diff = len(str(newUncert)) - len(str(newNom))
        if diff > 0:
            s = str(newNom) + int(diff)*'0' + '+/-' +str(newUncert)
        else:
            s = str(newValue)
    except:
        s = str(value)
    return s
