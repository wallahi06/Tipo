from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'The Tipo Library consists of great AI tools'

# Setting up
setup(
    name="Tipo",
    version=VERSION,
    author="Liam Nordvall",
    author_email="<liam_nordvall@hotmail.com>",
    description=DESCRIPTION,
    long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'deeplearning', 'AI', 'machine learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
