from math import ceil

class Table():
    """A class for generating tables of the measurment and parameter data
    associated with a bicycle. """

    def __init__(self, source, latex, bicycles):
        """Sets the basic attributes of the table.

        Parameters
        ----------
        source : string
            One of the parameter types: `Measured` or `Benchmark` for now.
        latex : boolean
            If true, the variable names will be formatted with LaTeX.
        bicycles : tuple or list of Bicycles
            Bicycle objects of which their parameters should appear in the
            generated table. The order of the bicycles determines the order in
            the table.

        """
        self.source = source
        self.bicycles = bicycles
        self.latex = latex
        # go ahead and calculate the base data, which sets allVariables and
        # tableData
        self.generate_variable_list()
        self.generate_table_data()

    def generate_variable_list(self):
        # generate a complete list of the variables
        allVariables = []
        try:
            for bicycle in self.bicycles:
                allVariables += bicycle.parameters[self.source].keys()
        except TypeError:
            allVariables += self.bicycle.parameters[self.source].keys()
        # remove duplicates and sort
        self.allVariables = sorted(list(set(allVariables)),
                key=lambda x: x.lower())

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

    def create_rst_table(self, fileName=None):
        """Returns a reStructuredText version of the table.

        Parameters
        ----------
        fileName : string
            If a path to a file is given, the table will be written to that
            file.

        Returns
        -------
        rstTable : string
            reStructuredText version of the table.

        """

        table = self.tableData

        # add the math directive if using latex
        if self.latex:
            for i, row in enumerate(table):
                self.tableData[i][0] = ':math:`' + row[0] + '`'

        # add a sub header
        table.insert(0, ['Variable'])
        for i, bicycle in enumerate(self.bicycles):
            if self.latex:
                table[0] += [':math:`v`', ':math:`\sigma`']
            else:
                table[0] += ['v', 'sigma']

        # find the longest string in each column
        largest = [0] # the top left is empty
        for bicycle in self.bicycles:
            l = int(ceil(len(bicycle.bicycleName) / 2.0))
            largest += [l, l]
        for row in table:
            colSize = [len(string) for string in row]
            for i, pair in enumerate(zip(colSize, largest)):
                if pair[0] > pair[1]:
                    largest[i] = pair[0]

        # build the rst table
        rstTable = '+' + '-' * (largest[0] + 2)

        for i, bicycle in enumerate(self.bicycles):
            rstTable += '+' + '-' * (largest[2 * i + 1] + largest[2 * i + 2] + 5)

        rstTable += '+\n|' + ' ' * (largest[0] + 2)

        for i, bicycle in enumerate(self.bicycles):
            rstTable += '| ' + bicycle.bicycleName + ' ' * (largest[2 * i + 1] +
                    largest[2 * i + 2] + 4 - len(bicycle.bicycleName))
        rstTable += '|\n'

        for j, row in enumerate(table):
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

        if fileName is not None:
            f = open(fileName, 'w')
            f.write(rstTable)
            f.close()

        return rstTable

def to_latex(var):
    """Returns a latex representation for a given variable string name.

    Parameters
    ----------
    var : string
        One of the variable names used in the bicycleparameters package.

    Returns
    -------
    latex : string
        A string formatting for pretty LaTeX math print.

    """

    latexMap = {'f': 'f',
                'w': 'w',
                'gamma': '\gamma',
                'g': 'g',
                'lcs': 'l_{cs}',
                'hbb': 'h_{bb}',
                'lsp': 'l_{sp}',
                'lst': 'l_{st}',
                'lamst': '\lambda_{st}',
                'whb': 'w_{hb}',
                'LhbF': 'l_{hbF}',
                'LhbR': 'l_{hbR}',
                'd': 'd',
                'l': 'l',
                'c': 'c',
                'lam': '\lambda',
                'xcl': 'x_{cl}',
                'zcl': 'z_{cl}',
                'ds1': 'd_{s1}',
                'ds3': 'd_{s3}'}
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
        elif var.startswith('I'):
            latex = var[0] + '_{' + var[1:] + '}'
        else:
            raise

    return latex

def uround(value):
    '''Returns a string representation of a value with an uncertainity which
    has been rounded to significant digits based on the uncertainty value.

    Parameters
    ----------
    value : ufloat
        A single ufloat.

    Returns
    -------
    s : string
        A rounded string representation of `value`.

    2.4563752289999+/-0.0003797273827

    becomes

    2.4564+/-0.0004

    This probably doesn't work for weird cases like large uncertainties.

    '''
    try:
    # grab the nominal value and the uncertainty
        nom = value.nominal_value
    except AttributeError:
        s = str(value)
    else:
        uncert = value.std_dev
        if abs(nom) < 1e-15:
            s = '0.0+/-0.0'
        else:
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
