import os

import yaml

from dso.utils import safe_merge_dicts


def get_base_config(task):
    # Load base config
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config_common.yaml"), encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Load task specific config
    task_config_file = None
    if task in ["regression", None]:
        task_config_file = "config_regression.yaml"
    else:
        # Custom tasks use config_common.yaml.
        task_config_file = "config_common.yaml"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), task_config_file), encoding="utf-8") as f:
        task_config = yaml.safe_load(f)

    return safe_merge_dicts(base_config, task_config)


def load_config(config=None):
    # Load user config
    if isinstance(config, str):
        with open(config, encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
    elif isinstance(config, dict):
        user_config = config
    else:
        assert config is None, "Config must be None, str, or dict."
        user_config = {}

    # Determine the task and language prior
    try:
        task = user_config["task"]["task_type"]
    except KeyError:
        task = "regression"
        print("WARNING: Task type not specified. Falling back to default task type '{}' to load config.".format(task))

    # Load task-specific base config
    base_config = get_base_config(task)

    # Return combined configs
    return safe_merge_dicts(base_config, user_config)
