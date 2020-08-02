from distutils.core import setup

setup(
    name='time_series_transform',
    version='v0',
    packages=[
        'time_series_transform',
        'time_series_transform/transform_core_api',
        'time_series_transform/stock_transform',
        'time_series_transform/test',
        ],
    license='MIT',
    long_description=open('README.md').read(),
)