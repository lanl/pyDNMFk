from setuptools import setup, find_packages
import pkg_resources

__version__ = "1.0"

# enforce pip version
pkg_resources.require(['pip >= 20.0.1'])

# add readme
with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# add dependencies
with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

setup(
    name='pyDNMFk',
    version=__version__,
    author='Ismael Boureima, Manish Bhattarai, Boian Alexandrov',
    author_email='iboureima@lanl.gov',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/lanl/pyDNMFk',  # change this to GitHub once published
    description='Distributed out-of-memory Nonnegative Matrix Factorization on CPU/GPU',
    setup_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'mpi4py', 'pytest-mpi', 'scikit-learn', 'pytest'],
    install_requires=INSTALL_REQUIRES,
    packages=['pyDNMFk'],
    python_requires='>=3.7.1',
    classifiers=[
        'Development Status :: ' + str(__version__) + ' - Beta',
        'Programming Language :: Python :: 3.7.1',
        'Topic :: Machine Learning :: Libraries'
    ],
    license='License :: BSD3 License',
    zip_safe=False
)
