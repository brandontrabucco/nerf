from setuptools import find_packages
from setuptools import setup


F = 'README.md'
with open(F, 'r') as readme:
    LONG_DESCRIPTION = readme.read()


CLASSIFIERS = [
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7']


setup(name='nerf-pytorch', version='1.2', license='MIT',
      packages=find_packages(include=['nerf', 'nerf.*']),
      description='Neural Radiance Fields',
      long_description=LONG_DESCRIPTION, classifiers=CLASSIFIERS,
      long_description_content_type='text/markdown',
      keywords=['Deep Learning', 'Neural Networks', "Vision"],
      author='Brandon Trabucco', author_email='brandon@btrabucco.com',
      url='https://github.com/brandontrabucco/nerf',
      download_url='https://github.com/brandontrabucco'
                   '/nerf/archive/v1_2.tar.gz')
