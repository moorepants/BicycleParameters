from distutils.core import setup

setup(
    name='BicycleParameters',
    version='0.1.0',
    author='Jason Keith Moore',
    author_email='moorepants@gmail.com',
    packages=['bicycleparameters', 'bicycleparameters.test'],
    scripts=['bin/create_bicycle_files.py','bin/largest_eigenvalue_plot.py'],
    url='http://pypi.python.org/pypi/BicycleParameters/',
    license='LICENSE.txt',
    description='Generates and manipulates the physical parameters of a bicycle.',
    long_description=open('README.rst').read(),
)
