from setuptools import setup, Extension, find_packages


setup(name='myspkmeans',
      version='1.0',
      description='spkmeans module',
      ext_modules=[Extension('myspkmeans', sources=['spkmeansmodule.c', 'spkmeans.c'])])
