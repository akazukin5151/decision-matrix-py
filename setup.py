import os
from pathlib import Path

from setuptools import setup

# The directory containing this file
HERE = Path(__file__).parent
README = (HERE / 'README.md').read_text()


setup(
    name='decision_matrix',
    version='0.1.0',
    description='Help you decide between choices with multiple criteria',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/twenty5151/decision-matrix',
    project_urls={
        'Source Code': 'https://github.com/twenty5151/decision-matrix',
        'Documentation': 'https://decision-matrix.readthedocs.io/en/latest/',
    },
    author='twenty5151',
    license='Mozilla Public License 2.0 (MPL 2.0)',
    classifiers=[
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Programming Language :: Python :: 3',
    ],
    packages=['matrix'],
    install_requires=[
        'pandas~=1.1',
        'numpy~=1.19',
        'matplotlib~=3.3',
        'scipy~=1.5',
    ],
    tests_require=['pytest~=6.0', 'hypothesis~=5.29'],
    extras_require={'cli': ['rich~=5.2', 'click~=7.1']},
    entry_points={
        'console_scripts': [
            'matrix=matrix.cli:main',
        ]
    },
)
