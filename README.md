# pc-autoencoder
Tensorflow code for optimal binary autoencoding using pairwise correlations between encoded and decoded bits, as described in: 
> Optimal Binary Autoencoding with Pairwise Correlations. Akshay Balsubramani. ICLR 2017. [Link to arXiv](http://arxiv.org/abs/1611.02268)



## Installation instructions

Create and activate a conda environment first with 

`conda create -n my-env-name python=3.8.5`
`conda activate my-env-name`

Install a bunch of packages:

`pip install -r requirements.txt`

Finally, from within the activated environment, it's important to install a new Jupyterlab kernel:
`python -m ipykernel install --user --name=my-kernel-name`

Now, when you run
`jupyter lab`
from inside the environment, you'll be given the option to run any notebook using kernel "my-kernel-name" instead of "Python 3".


(Note: you may want to first set `conda config --set channel_priority strict` to avoid typing `-c conda-forge` with each package installation).