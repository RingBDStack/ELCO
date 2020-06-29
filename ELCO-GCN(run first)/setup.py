from setuptools import setup
from setuptools import find_packages

setup(name='gcn',
      version='1.0',
      description='Graph Convolutional Networks in Tensorflow',
      install_requires=['numpy>=1.15.4',
                        'tensorflow>=1.15.2,<2.0',
                        'networkx>=2.2',
                        'scipy>=1.1.0',
                        'scikit-learn>=0.22.2.post1',
                        'karateclub>=1.0.1'
                        ],
      package_data={'gcn': ['README.md']},
      packages=find_packages())
