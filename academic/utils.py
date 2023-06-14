import random
import torch
import numpy as np
import warnings
from collections import OrderedDict


def load_model(model, path):
    parameter_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in parameter_dict.items():  # k is module.xxx.weight, v is weight
        name = k[7:]  # truncate the xxx.weight after `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)  # load model
    return model


def reset_weights(model_child):
    """
    Use to recursively reset weights in the model
    """

    @torch.no_grad()
    def apply(m):
        for name, child in m.named_children():
            # if isinstance(child, CountReLU):
            #     child.stats = None
            if hasattr(child, "_parameters"):
                for param_name in child._parameters:
                    # print(name, param_name)
                    if child._parameters[param_name] is not None:
                        if len(child._parameters[param_name].shape) < 2:
                            torch.nn.init.normal_(child._parameters[param_name].data)
                        else:
                            torch.nn.init.xavier_uniform_(child._parameters[param_name].data)
            reset_parameters = getattr(child, "reset_parameters", None)
            if callable(reset_parameters):
                child.reset_parameters()
            else:
                apply(child)

    apply(model_child)
    if hasattr(model, "__init_weights__"):
        model_child.__init_weights__()


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
