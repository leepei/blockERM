This directory includes sources used in the following paper:

Ching-pei Lee and Kai-Wei Chang, Distributed Block-diagonal Approximation Methods for Regularized Empirical Risk Minimization, 2017.
You will be able to regenerate experiment results in the paper using this implementation.
However, results may be slightly different due to the randomness, the CPU speed,
and the load of your computer.

Please cite the above article if you find this tool useful. Please also read
the COPYRIGHT before using this tool.

The implementation is based on MPI LIBLINEAR.

If you have any questions, please contact:
Ching-pei Lee
ching-pei@cs.wisc.edu

Solvers
=======
The following solvers are supported.
-	0 -- Block-diagonal approximation approach proposed in this paper
-	1 -- Trust region Newton method for L2-loss SVM (primal)
-	2 -- DisDCA practical variant/CoCoA+


Solver 1 is directly copied from MPI LIBLINEAR. Its documentation is available in
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/
Zhuang, Yong, Chin, Wei-Sheng, Juan, Yu-Chin, and Lin, Chih-Jen. Distributed Newton method for regularized logistic regression. In Proceedings of The Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), 2015.

Solver 0 implements the method in
Ching-pei Lee and Kai-Wei Chang, Distributed Block-diagonal Approximation Methods for Regularized Empirical Risk Minimization, 2017.

Solver 2 implements the DisDCA practical variant in
Yang, Tianbao. Trading computation for communication: Distributed stochastic dual coordinate ascent. In Advances in Neural Information Processing Systems 26, pp. 629–637, 2013.

Problems
========
The following L2-regularized problems are supported.
- 0 -- L1-loss SVM
- 1 -- L2-loss SVM
- 2 -- Logistic regression


Running Environment
===================
This code is supposed to be run on UNIX machines. The following
commands are required:

- g++
- make
- split

All methods require MPI libraries. Available implementations include

- OpenMPI
You can find the information about OpenMPI at

http://www.open-mpi.org/

- MPICH
You can find the information about MPICH at

http://www.mpich.org/

- MVAPICH
You can find the information about MVAPICH at

http://mvapich.cse.ohio-state.edu

Usage
=====
To compile the code, type

	> make

To train a model distributedly, we provide a script "split.py" to split data into segments.
This scripts requires a file of the node list, and one core is treated as one node.
Namely, if you wish to use n > 1 cores on one machine, just duplicate this machine on the list n times.
We provide an example of the node list in the file "machinelist"
Assume the original training file name is trainfile,
then the segmented filenames will be trainfile.00,
trainfile.1,...
The enumeration starts from 0, and all files are of the same number of digits for enumeration.
To see the full usage of this script, type

	> python split.py

The segmented files will then be copied to the machines


To run the package, type

	> mpirun -hostfile machinelist ./train

and see the usage.

