# from ray import tune
HYPERPARAM_CONFIGS = {
    "StateSpace2dJoints":{
        # "lr": tune.loguniform(1e-4, 1e-1),
        # "encoder_layers": [tune.choice([32, 64])],
        # "n_cells": tune.choice([1, 2]),
        # "rnn_dim": tune.choice([32, 64, 128, 256]),
        # "batch_size": tune.choice([32, 64]),

        "method": "random",
        "metric": {
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "lr": {
                "distribution": "log_uniform",
                "min": 1e-5,
                "max": 1e-2
            },
            "encoder_layers":{
                "values": [32, 64, 128]
            },
            "n_cells":{
                "values": [1, 2]
            },
            "rnn_dim":{
                "values": [32, 64, 128, 256]
            },
            "batch_size":{
                "values": [32, 64, 128]
            },

        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3

        }       
    }
}