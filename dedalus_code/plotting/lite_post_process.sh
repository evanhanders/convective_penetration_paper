#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
DIR=""
NCORE=""

while getopts ":d:n:h?:" opt; do
    case "$opt" in
    h|\?)
        echo "specify dir with -d and core number with -n" 
        exit 0
        ;;
    d)  DIR=$OPTARG
        ;;
    n)  NCORE=$OPTARG
        ;;
    esac
done
echo $DIR
echo $NCORE
echo "Processing $DIR on $NCORE cores"

mpiexec_mpt -n $NCORE python3 find_top_cz.py $DIR
mpiexec_mpt -n $NCORE python3 theory_movie.py $DIR
mpiexec_mpt -n $NCORE python3 plot_avg_profiles.py $DIR

cd $DIR
tar -cvf avg_profs.tar avg_profs/
cd $OLDPWD
