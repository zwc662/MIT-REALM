# Simplex-Cartpole

This repo investigates the Simplex-enabled Safe Continual Learning Machine for Cartpole.

### Environment

This project is using the following settings:

- Ubuntu: 22.04
- python: 3.9.5

### Package in python

```
pip install -r requirement.txt
```

### Package in Matlab

Download and install [CVX](http://cvxr.com/cvx/) package. To solve the LMIs optimization problem to get the
physical-model-based controller and P-matrix, you can run the solve_lmis.m file in Matlab

## Run

```
main_ips.py [-h] [--config CONFIG] [--generate_config] [--force]
                  [--params [PARAMS [PARAMS ...]]] [--mode MODE]
                  [--gpu] [--id RUN_ID][--weights PATH_TO_WEIGHTS] 

arguments:
  --config             Specifying different configuration .json files for different test.
  --generate_config    Generating default configuration .json files. 
  --force              Over-writting the previous run of same ID.
  --params             Over-writting params setting.
  --gpu                Enabling gpu device to speed up training. Training using CPU if not specified.   
  --mode MODE          Training or testing [train|test]
  --id                 Assigning an ID for the training/testing.
  --weights            Loading pretrained weights.    
```

Example_1: Generate configuration file

```
python main_ips.py --generate_config
```

Example_2: Training/Testing

```
python main_ips.py --config {PATH_TO_CONFIG_FILE} --mode {train|test} --id {RUN_NAME} --gpu --weights {PATH_TO_PRETRAINED_WEIGHTS}
```

Example_3: evaluation

```
python evalutation/eval_xxx.py --config {PATH_TO_CONFIG_FILE} --id {RUN_NAME} --weights {PATH_TO_PRETRAINED_WEIGHTS}
```

