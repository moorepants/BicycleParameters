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
    install_requires=['numpy>=1.6.1',
                      'scipy>=0.9.0',
                      'matplotlib>=1.1.1',
                      'uncertainties>=2.0.0',
                      'yeadon>=1.1.0',
                      'DynamicistToolKit>=0.1.0'],
    extras_require={'doc': ['sphinx', 'numpydoc']},
    tests_require=['nose'],
    test_suite='nose.collector',
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Operating System :: OS Independent',
                 'Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics']
)
