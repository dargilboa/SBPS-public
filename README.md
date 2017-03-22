# SBPS-public

This is an implementation of the Stochastic Bouncy Particle Sampler described in https://arxiv.org/abs/1609.00770, a stochastic gradient based MCMC sampler that implements a piecewise deterministic Markov process.

Usage instructions: pSBPS_fig.ipynb can be used to compare SBPS and the preconditioned variant pSBPS when sampling from a highly anisotropic logistic regression posterior.
The sampler code itself is found in samplers.py, while various useful functions for running the samplers are in utils.py

