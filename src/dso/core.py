import os
import zlib
import random

from collections import defaultdict
from datetime import datetime
from time import time

import numpy as np
import torch
import yaml

from dso.task import set_task
from dso.train import Trainer
from dso.train_stats import StatsLogger
from dso.prior import make_prior
from dso.program import Program
from dso.config import load_config
from dso.state_manager import make_state_manager
from dso.policy.policy import make_policy
from dso.policy_optimizer import make_policy_optimizer


class DeepSymbolicOptimizer:
    """
    Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.

    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON.

    Attributes
    ----------
    config : dict
        Configuration parameters for training.

    Methods
    -------
    train
        Builds and trains the model according to config.
    """

    def __init__(self, config=None):
        self.set_config(config)
        self.is_first_step = True

    def setup(self):
        # Clear the cache and reset the compute graph
        Program.clear_cache()

        # Generate objects needed for training and set seeds
        self.make_task()
        self.set_seeds()  # Must be called _after_ resetting graph and _after_ setting task

        # Setup logdirs and output files
        self.output_file = self.make_output_file()
        self.save_config()

        # Prepare training parameters
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.policy = self.make_policy()
        self.policy_optimizer = self.make_policy_optimizer()
        self.gp_controller = self.make_gp_controller()
        self.logger = self.make_logger()
        self.trainer = self.make_trainer()

    def train_one_step(self, override=None):
        """
        Train one iteration.
        """

        # Setup the model
        if self.is_first_step:
            self.setup()
            self.is_first_step = False

        # Run one step
        assert not self.trainer.done, "Training has already completed!"
        self.trainer.run_one_step(override)

        # If complete, return summary
        if self.trainer.done:
            return self.finish()

    def train(self):
        """
        Train the model until completion.
        """

        # Setup the model
        self.setup()

        # Train the model until done
        while not self.trainer.done:
            result = self.train_one_step()

        return result

    def finish(self):
        """
        After training completes, finish up and return summary dict.
        """

        # Return statistics of best Program
        p = self.trainer.p_r_best
        result = {"seed": self.config_experiment["seed"]}  # Seed listed first
        result.update({"r": p.r})
        result.update(p.evaluate)
        result.update({"expression": repr(p.sympy_expr), "traversal": repr(p), "program": p})

        # Save all results available only after all iterations are finished. Also return metrics to be added to the summary file
        results_add = self.logger.save_results(self.trainer.nevals)
        result.update(results_add)

        return result

    def set_config(self, config):
        config = load_config(config)
        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_logger = self.config["logging"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_policy = self.config["policy"]
        self.config_policy_optimizer = self.config["policy_optimizer"]
        self.config_gp_meld = self.config["gp_meld"]
        self.config_experiment = self.config["experiment"]

    def save_config(self):
        # Save the config file
        if self.output_file is not None:
            path = os.path.join(self.config_experiment["save_path"], "config.yaml")
            # With run.py, config.yaml may already exist. To avoid race
            # conditions, only record the starting seed. Use a backup seed
            # in case this worker's seed differs.
            backup_seed = self.config_experiment["seed"]
            if not os.path.exists(path):
                if "starting_seed" in self.config_experiment:
                    self.config_experiment["seed"] = self.config_experiment["starting_seed"]
                    del self.config_experiment["starting_seed"]
                with open(path, "w") as f:
                    yaml.safe_dump(dict(self.config), f)
            self.config_experiment["seed"] = backup_seed

    def set_seeds(self):
        """
        Set the torch, numpy, and random module seeds based on the seed
        specified in config. If there is no seed or it is None, a time-based
        seed is used instead and is written to config.
        """

        seed = self.config_experiment.get("seed")

        # Default uses current time in milliseconds, modulo 1e9
        if seed is None:
            seed = round(time() * 1000) % int(1e9)
            self.config_experiment["seed"] = seed

        # Shift the seed based on task name
        # This ensures a specified seed doesn't have similarities across different task names
        task_name = Program.task.name
        shifted_seed = seed + zlib.adler32(task_name.encode("utf-8"))

        # Set the seeds using the shifted seed
        torch.manual_seed(shifted_seed)
        np.random.seed(shifted_seed)
        random.seed(shifted_seed)

    def make_prior(self):
        prior = make_prior(Program.library, self.config_prior)
        return prior

    def make_state_manager(self):
        state_manager = make_state_manager(self.config_state_manager)
        return state_manager

    def make_trainer(self):
        trainer = Trainer(self.policy, self.policy_optimizer, self.gp_controller, self.logger, **self.config_training)
        return trainer

    def make_logger(self):
        logger = StatsLogger(self.output_file, **self.config_logger)
        return logger

    def make_policy_optimizer(self):
        policy_optimizer = make_policy_optimizer(self.policy, **self.config_policy_optimizer)
        return policy_optimizer

    def make_policy(self):
        policy = make_policy(self.prior, self.state_manager, **self.config_policy)
        return policy

    def make_gp_controller(self):
        if self.config_gp_meld.pop("run_gp_meld", False):
            from dso.gp.gp_controller import GPController

            gp_controller = GPController(self.prior, self.config_prior, **self.config_gp_meld)
        else:
            gp_controller = None
        return gp_controller

    def make_task(self):
        # Set the complexity function
        complexity = self.config_training["complexity"]
        Program.set_complexity(complexity)

        # Set the constant optimizer
        const_optimizer = self.config_training["const_optimizer"]
        const_params = self.config_training["const_params"]
        const_params = const_params if const_params is not None else {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        # Set the Task for the parent process
        set_task(self.config_task)

    def make_output_file(self):
        """Generates an output filename"""

        # If logdir is not provided (e.g. for pytest), results are not saved
        if self.config_experiment.get("logdir") is None:
            self.save_path = None
            print("WARNING: logdir not provided. Results will not be saved to file.")
            return None

        # When using run.py, timestamp is already generated
        timestamp = self.config_experiment.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.config_experiment["timestamp"] = timestamp

        # Generate save path
        task_name = Program.task.name
        if self.config_experiment["exp_name"] is None:
            save_path = os.path.join(self.config_experiment["logdir"], "_".join([task_name, timestamp]))
        else:
            save_path = os.path.join(self.config_experiment["logdir"], self.config_experiment["exp_name"])

        self.config_experiment["task_name"] = task_name
        self.config_experiment["save_path"] = save_path
        os.makedirs(save_path, exist_ok=True)

        seed = self.config_experiment["seed"]
        output_file = os.path.join(save_path, "dso_{}_{}.csv".format(task_name, seed))

        self.save_path = save_path

        return output_file
