from setuptools import setup, find_packages

setup(
    name='udmurt_postagger',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'udmurt_postagger': ['resources/*']
    },
    install_requires=[
        'tensorflow>=2.0.0',
        'numpy',
    ],
    description='Udmurt POS Tagger Library',
    author='codemurt',
    author_email='egor.lebe@inbox.ru',
)