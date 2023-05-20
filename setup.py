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
    version='0.1.0',
    author='Esteban Ponce de Leon',
    description='A Python OSINT tool using Large Language Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/estebanpdl/osintgpt',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
        'Programming Language :: Python :: 3.8'
        'Programming Language :: Python :: 3.9'
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Intended Audience :: OSINT, Science/Research, Journalism, Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security :: Cryptography :: OSINT'
    ],
    python_requires='>=3.11.2',
    install_requires=[]
)
