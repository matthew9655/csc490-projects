# Improved Tracker

We chose to tackle the occulusion handling problem. The class for the occlusion handler can be found in `improved.py`. We also use a number of helper classes to aid the occulusion handler. These can be found in `improved_types.py`.

## Running the Code

To do a hyperparameter sweep, you can either run `./run-serial.sh` or `./run_parallel.sh` if you have gnu-parallel.

### Serial

you can add extra hyperparameters into the respective lists in `run_serial.sh`. Furthermore, you can edit the filename to where the evaluations from the sweep will be placed.

### Parallel

you can add extra hyperparameters into the gnu-parallel lists in `run_parallel.sh`.

| command |       description        | type    |
| :-----: | :----------------------: | ------- |
|   $1    |       degree_thres       | boolean |
|   $2    |       score_thres        | int     |
|   $3    | frame difference divider | float   |
|   $4    | path to add evaluations  | string  |

## Extra args added

|    command     |                           description                            | type      |
| :------------: | :--------------------------------------------------------------: | --------- |
|   --improved   |                True to activate occlusion handler                | boolean   |
| --degree_thres |           minimmum yaw difference to find intersection           | int       |
| --score_thres  |        maximum score to not match a single-frame tracklet        | float     |
|   --divider    | affects the effect the frame difference has on the cost function | int/float |
|  --score_func  |                      cost function C1 or C2                      | 0 or 1    |

## Finding the best model

run `scan_results.py {path to evaluation results folder}`. This give stats on the model with the best MOTA / MOTP.
