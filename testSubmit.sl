#!/bin/bash

#SBATCH --cpus-per-task   8
#SBATCH --mem		30G
#SBATCH --time 		02:00:00


# example of submitting to cluster.

module load MATLAB/2023a

. ~/.bashrc # Only callum has to do this.

matlab -batch "mddPlot('/home/cwal219/app_examples/Delft3D/jon')"