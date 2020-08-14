import os
from distutils.core import setup

with open('requirement.txt') as f:
    required = f.read().splitlines()

setup(
    name='time_series_transform',
    version='0.0.1',
    packages=[
        'time_series_transform',
        'time_series_transform/transform_core_api',
        'time_series_transform/stock_transform',
        'time_series_transform/test',
        ],
    license='MIT',
    author_email = 'kuanlun.chiang@outlook.com',
    url = 'https://github.com/allen-chiang/Time-Series-Transformer',
    download_url ='https://github.com/allen-chiang/Time-Series-Transformer/archive/v0.0.0.tar.gz',
    keywords = ['time series','stock', 'machine learning', 'deep learning'],
    install_requires=required,
    author = 'KuanLun Chiang; KuanYu Chiang',
    long_description=open('README.md').read(),
    classifiers=[
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',      
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
