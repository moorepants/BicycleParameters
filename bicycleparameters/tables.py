from uncertainties import ufloat

def to_latex(var):
    latexMap = {'f': 'f',
                'w': 'w',
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
                'd': 'd',
                'l': 'l'}
    try:
        latex = latexMap[var]
    except KeyError:
        if var.startswith('alpha'):
            latex = r'\alpha_{' + var[-2:] + '}'
        elif var.startswith('a') and len(var) == 3:
            latex = 'a_{' + var[-2:] + '}'
        elif var.startswith('T'):
            latex = 'T^' + var[1] + '_{' + var[-2:] + '}'
        elif len(var) == 2:
            latex = var[0] + '_' + var[1]
        else:
            raise

    return latex

class Table():
    def __init__(self, source, latex, *bicycles):
        self.source = source
        self.bicycles = bicycles
        self.latex = latex
        self.generate_variable_list()
        self.generate_table_data()

    def generate_variable_list(self):
        # generate a complete list of the variables
        allVariables = []
        for bicycle in self.bicycles:
            allVariables += bicycle.parameters[self.source].keys()
        # remove duplicates and sort
        self.allVariables = sorted(list(set(allVariables)))

    def generate_table_data(self):
        """Generates a list of data for a table."""
        table = []
        for var in self.allVariables:
            # add a new line
            table.append([])
            if self.latex:
                table[-1].append(to_latex(var))
            else:
                table[-1].append(var)
            for bicycle in self.bicycles:
                try:
                    val, sig = uround(bicycle.parameters[self.source][var]).split('+/-')
                except ValueError:
                    val = str(bicycle.parameters[self.source][var])
                    sig = 'NA'
                except KeyError:
                    val = 'NA'
                    sig = 'NA'
                table[-1] += [val, sig]
        self.tableData = table

    def create_rst_table(self):

        if self.latex:
            for i, row in enumerate(self.tableData):
                self.tableData[i][0] = ':math:`' + row[0] + '`'

        # find the longest string in each column
        largest = [0]
        for bicycle in self.bicycles:
            l = len(bicycle.bicycleName) / 2
            largest += [l, l]
        for row in self.tableData:
            colSize = [len(string) for string in row]
            for i, pair in enumerate(zip(colSize, largest)):
                if pair[0] > pair[1]:
                    largest[i] = pair[0]

        # build the rst table
        rstTable = '+' + '-' * (largest[0] + 2)

        for i, bicycle in enumerate(self.bicycles):
            rstTable += '+' + '-' * (largest[i + 1] + largest[i + 2] + 5)
        rstTable += '+\n|' + ' ' * (largest[0] + 2)

        for i, bicycle in enumerate(self.bicycles):
            rstTable += '| ' + bicycle.bicycleName + ' ' * (largest[i + 1] +
                    largest[i + 2] + 4 - len(bicycle.bicycleName))
        rstTable += '|\n'

        for j, row in enumerate(self.tableData[1:]):
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
    except AttributeError:
        s = str(value)
    else:
        uncert = value.std_dev()
        # convert the uncertainty to a string
        s = '%.14f' % uncert
        # find the first non-zero character
        for j, number in enumerate(s):
            if number == '0' or number == '.':
                pass
            else:
                digit = j
                break
        newUncert = round(uncert, digit - 1)
        newUncertStr = ('%.' + str(digit - 1) + 'f') % newUncert
        newNom = round(nom, len(newUncertStr) - 2)
        newNomStr = ('%.' + str(digit - 1) + 'f') % newNom
        diff = len(newUncertStr) - len(newNomStr)
        if diff > 0:
            s = newNomStr + int(diff) * '0' + '+/-' + newUncertStr
        else:
            s = newNomStr + '+/-' + newUncertStr
    return s
