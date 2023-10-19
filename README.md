# Q-EXP

[Bayesian Learning via Q-Exponential Process](https://arxiv.org/abs/2210.07987), NIPS 2023

Shuyi Li, Michael O'Connor and Shiwei Lan <slan@asu.edu>


1. util - class definitions of different priors
	* gp: Gaussian process
	* bsv: Besov process
	* qep: Q-EP process
	* epp: not a process, similar class based on Gomez's EP distribution
	* kernel: class of kernel including subclasses:
		+ covf: covariance function
		+ serexp: series expansion
		+ graphL: graph Laplacian
2. demo - demo codes for testing 
	* test_inconsistency.py: the inconsistency of Gomez's EP distribution
	* test_consistency.py: the consistency of our proposed Q-ED distribution
	* test_wnrep.py: the equivalence between the stochastic representation and the white-noise representation
3. satellite - files to reproduce the MAP reconstruction of blurry image of satellite
	* misfit.py: definition of the likelihood model
	* prior.py: definition of prior model
	* Satellite.py: definition of the linear inverse problem for image reconstruction

You can run:

test_xxx.py to numerically verify the corresponding facts

Satellite.py to generate the MAP reconstructions.