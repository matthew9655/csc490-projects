#!/bin/bash
parallel -j 22 ./script_parallel.sh {1} {2} {3} {4} ::: 60 ::: 1.5 2.5 ::: 4 6 ::: results