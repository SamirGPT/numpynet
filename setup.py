from setuptools import setup, find_packages

setup(
    name='numpynet',
    version='1.1.0',
    description='A deep learning library in pure NumPy',
    author='SamirGPT & Manus',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
