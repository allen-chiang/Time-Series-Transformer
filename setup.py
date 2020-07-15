from distutils.core import setup

setup(
    name='time_series_transform',
    version='v0',
    packages=['time_series_transform','time_series_transform/stock_transform'],
    license='MIT',
    long_description=open('README.md').read(),
)