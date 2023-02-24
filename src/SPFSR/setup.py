from setuptools import setup, find_packages

setup(name='spfsr',
      version='0.1',
      description='Sequentially-Projected Fast Sparse Regression algorithm using parabolic interpolation',
      url='https://github.com/erirosv/SPFSR',
      author='Erik Rosvall',
      author_email='erirosv@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'scikit-learn'
      ],
      zip_safe=False)
