import torch
from collections import OrderedDict


def load_model(model, path):
    parameter_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in parameter_dict.items(): # k is module.xxx.weight, v is weight
            name = k[7:] # truncate the xxx.weight after `module.`
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict) # load model
    return model


def reset_weights(model_child):
        """
        Use to recursively reset weights in the model
        """
        @torch.no_grad()
        def apply(m):
            for name, child in m.named_children():
                if isinstance(child, CountReLU):
                    child.stats = None
                if hasattr(child, "_parameters"):
                    for param_name in child._parameters:
                        # print(name, param_name)
                        if child._parameters[param_name] is not None:
                            if len(child._parameters[param_name].shape) < 2:
                                torch.nn.init.normal_(
                                    child._parameters[param_name].data
                                )
                            else:
                                torch.nn.init.xavier_uniform_(
                                    child._parameters[param_name].data
                                )
                reset_parameters = getattr(child, "reset_parameters", None)
                if callable(reset_parameters):
                    child.reset_parameters()
                else:
                    apply(child)
        apply(model_child)
        if hasattr(model, "__init_weights__"):
            model_child.__init_weights__()
    