#!/bin/bash
parallel -j 20 ./python_script.sh {1} {2} {3} ::: 60 70 80 90 100 110 ::: 1.5 2.0 2.5 3.0 3.5 4.0 ::: 3 4 5 6