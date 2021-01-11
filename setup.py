import os
from setuptools import setup

required = [
    'numpy',
    'pandas',
    'pyarrow',
    'yfinance',
    'scipy',
    'PyWavelets',
    'Sphinx',
    'rinohtype',
    'nbsphinx',
    'pytest-dependency',
    'scipy',
    'matplotlib',
    'sklearn',
    'joblib'
]

with open("README.md", "r") as fh:
    long_description = fh.read() 

setup(
    name='time_series_transform',
    version='1.1.3',
    description = 'A data preprocessing package for time series data. It provides tools for machine learing and deep learning.It also has subsection specialized in stock.',
    packages=[
        'time_series_transform',
        'time_series_transform/transform_core_api',
        'time_series_transform/plot',
        'time_series_transform/stock_transform',
        'time_series_transform/stock_transform/stock_engine',
        'time_series_transform/test',
        'time_series_transform/io'
        ],
    license='MIT',
    author_email = 'kuanlun.chiang@outlook.com',
    url = 'https://github.com/allen-chiang/Time-Series-Transformer',
    project_urls = {
        'Source Code' : 'https://github.com/allen-chiang/Time-Series-Transformer',
        'Documentation' : "https://allen-chiang.github.io/Time-Series-Transformer/"
    },
    download_url ='https://github.com/allen-chiang/Time-Series-Transformer/archive/1.1.2.tar.gz',
    keywords = ['time series','stock', 'machine learning', 'deep learning'],
    install_requires = required,
    author = 'Kuan-Lun Chiang; Kuan-Yu Chiang',
    long_description= long_description,
    long_description_content_type='text/markdown',
    classifiers=[
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',      
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
