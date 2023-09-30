# import modules
from setuptools import setup, find_packages

# get README.md file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# install requires > read requirements.txt
with open('./requirements.txt', 'r', encoding='utf-8') as f:
    install_requires = f.read().splitlines()

# setup package
setup(
    name='osintgpt',
    version='0.1.23',
    author='Esteban Ponce de Leon',
    description='A Python OSINT tool using Large Language Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/estebanpdl/osintgpt',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: General',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Utilities'
    ],
    python_requires='>=3.7',
    install_requires=install_requires
)
