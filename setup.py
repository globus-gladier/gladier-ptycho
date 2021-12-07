import os
from setuptools import setup, find_packages
from glob import glob 
# single source of truth for package version
version_ns = {}
with open(os.path.join('gladier_pytcho', 'version.py')) as f:
    exec(f.read(), version_ns)
version = version_ns['__version__']

install_requires = []
with open('requirements.txt') as reqs:
    for line in reqs.readlines():
        req = line.strip()
        if not req or req.startswith('#'):
            continue
        install_requires.append(req)

script_list = glob('scripts/xpcs_*')

setup(
    name='gladier_ptycho',
    description='Ptychography Gladier client',
    url='https://github.com/globus-gladier/gladier-Pytchp',
    maintainer='Tekin Bicer',
    maintainer_email='',
    version=version,
    scripts=script_list,
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=[],
    license='Apache 2.0',
    classifiers=[]
)
