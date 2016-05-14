#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector Quantization coding
===================
A file compactor
"""
from setuptools import setup, find_packages

install_requires = [
    'bitarray>=0.8.1',
    'ipdb>=0.9.0',
    'optparse-pretty>=0.1.1',
    'bintrees>=2.0.2',
    'scipy>=0.17.0',
    'numpy>=1.11.0',
]


setup(
    name="lbg",
    version='0.0.8',
    author='Luiz Oliveira',
    author_email='ziuloliveira@gmail.com',
    url='https://github.com/Ziul/lbg/',
    entry_points={
        'console_scripts': [
            'lbg = lbg:main',
            # 'main-test = main:test',
        ]},
    description='A program to compact/descompact a file',
    long_description=__doc__,
    license='GPLv3',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    test_suite="tests.run.runtests",
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3',
        'Topic :: Utilities',
    ],
)
