#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('bicycleparameters/version.py').read())

setup(
    name='BicycleParameters',
    version=__version__,
    author='Jason Keith Moore',
    author_email='moorepants@gmail.com',
    packages=find_packages(),
    url='http://pypi.python.org/pypi/BicycleParameters',
    license='LICENSE.txt',
    description='Generates and manipulates the physical parameters of a bicycle.',
    long_description=open('README.rst').read(),
    install_requires=[
        'DynamicistToolKit>=0.5.3',
        'matplotlib>=3.1.2',
        'numpy>=1.17.4',
        'pyyaml>=5.3.1',
        'scipy>=1.3.3',
        'uncertainties>=3.1.2',
        'yeadon>=1.3.0',
    ],
    extras_require={
        'doc': ['sphinx', 'numpydoc'],
        'app': ['dash>=2', 'dash-bootstrap-components'],
    },
    tests_require=['nose'],
    test_suite='nose.collector',
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Operating System :: OS Independent',
                 'Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics']
)
