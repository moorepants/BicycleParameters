#!/usr/bin/env python

from setuptools import setup, find_packages

exec(open('bicycleparameters/version.py').read())

setup(
    name='BicycleParameters',
    version=__version__,
    author='Jason K. Moore',
    author_email='moorepants@gmail.com',
    packages=find_packages(),
    url='https://bicycleparameters.readthedocs.io',
    license='LICENSE.txt',
    description='Generates and manipulates the physical parameters of a bicycle.',
    long_description=open('README.rst').read(),
    project_urls={
        'Documentation': 'http://bicycleparameters.readthedocs.io',
        'Issue Tracker': 'https://github.com/moorepants/BicycleParameters/issues',
        'Source Code': 'https://github.com/moorepants/BicycleParameters',
        'Web Application': 'https://bicycle-dynamics.onrender.com',
    },
    include_package_data=True,  # includes things in MANIFEST.in
    install_requires=[
        'DynamicistToolKit>=0.5.3',
        'matplotlib>=3.5.1',
        'numpy>=1.21.5',
        'pyyaml>=5.4.1',
        'scipy>=1.8.0',
        'uncertainties>=3.1.5',
        'yeadon>=1.3.0',
    ],
    python_requires='>=3.9',
    extras_require={
        'doc': [
            'numpydoc>=1.2',
            'sphinx-gallery',
            'sphinx-reredirects',
            'sphinx>=4.3.2',
        ],
        'app': [
            'dash-bootstrap-components',
            'dash>=2',
            'pandas>=1.3.5'
        ],
    },
    entry_points={'console_scripts':
                  ['bicycleparameters = bicycleparameters.app:app.run_server']},
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 'Programming Language :: Python :: 3.12',
                 'Programming Language :: Python :: 3.13',
                 'Operating System :: OS Independent',
                 'Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Natural Language :: English',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics']
)
