from setuptools import setup, find_packages

setup(name='Feature Selection',
      version='0.1',
      description='Feature selection code',
      url='https://github.com/erirosv/FeatureSelection',
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