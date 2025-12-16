import numpy as np
from omegaconf import OmegaConf

from dacboenv.task import get_dacbo_task_name, get_perceptron_configspace

OmegaConf.register_new_resolver(name="get_perceptron_configspace", resolver=get_perceptron_configspace, replace=True)
OmegaConf.register_new_resolver(name="get_dacbo_task_name", resolver=get_dacbo_task_name, replace=True)
OmegaConf.register_new_resolver(name="len", resolver=lambda x: len(x), replace=True)
OmegaConf.register_new_resolver(name="multiply", resolver=lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver(name="add", resolver=lambda x, y: x + y, replace=True)
OmegaConf.register_new_resolver(name="divide", resolver=lambda x, y: x / y, replace=True)
OmegaConf.register_new_resolver(name="divideint", resolver=lambda x, y: x // y, replace=True)
OmegaConf.register_new_resolver(name="range", resolver=lambda x: list(range(x)), replace=True)
OmegaConf.register_new_resolver(name="zeros", resolver=lambda x: list(np.zeros(x)), replace=True)
OmegaConf.register_new_resolver(name="emptylist", resolver=list, replace=True)
OmegaConf.register_new_resolver(
    name="get_instance_features", resolver=lambda x: {v: [i] for i, v in enumerate(x)}, replace=True
)
