# Gladier Ptychography Application

The Ptychography flow extends the [APWF](https://github.com/tekinbicer/apwf) workflow to perform reconstruction of ptychography datasets generated at Advanced Photon Source (APS), Argonne National Laboratory (ANL).

The flow uses remote, large-scale compute resources (or supercomputers) via Globus toolkits; namely, funcX, transfer and flows. All tools rely on Globus authentication infrastructure (Auth) hence providing single point of sign-on for execution of data-intensive workflows on supercomputers.

APWF, and this application, use Tike toolbox to perform parallel ptychographic reconstruction. Tike is optimized for multi-node and multi-GPU settings and can perform efficient reconstruction on high-end GPU resources.

## Running

The ptychography flow uses a shell command tool to execute the [Ptychodus](https://github.com/AdvancedPhotonSource/ptychodus) tool on the example data.

Your compute endpoint will require the following dependencies:

```
#Ptycho tools
git clone https://github.com/AdvancedPhotonSource/ptychodus
cd ptychodus
conda install -c conda-forge --file requirements-dev.txt
conda install -c conda-forge tike
pip install -e . 
```

```bash

python gladier_ptychodus/ptychodus_client.py --datadir <data path> --localdir <data path>
```