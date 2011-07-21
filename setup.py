from distutils.core import setup

setup(
    name='BicycleParameters',
    version='0.1.0dev',
    author='Jason Keith Moore',
    author_email='moorepants@gmail.com',
    packages=['bicycleparameters', 'bicycleparameters.test'],
    url='http://github.com/moorepants/BicycleParameters/',
    license='LICENSE.txt',
    description='Generates and manipulates the physical parameters of a bicycle.',
    long_description=open('README.rst').read(),
)
