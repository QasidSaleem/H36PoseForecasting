from ray import tune
HYPERPARAM_CONFIGS = {
    "StateSpace2dJoints":{
        "lr": tune.loguniform(1e-4, 1e-1),
        "encoder_layers": [tune.choice([32, 64])],
        "n_cells": tune.choice([1, 2]),
        "rnn_dim": tune.choice([32, 64, 128, 256]),
        "batch_size": tune.choice([32, 64]),
             
    }
}