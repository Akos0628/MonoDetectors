import os
import torch

def load_checkpoint(model, optimizer, filename, map_location):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=map_location, weights_only = False)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint.get('model_state', None) is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint.get('optimizer_state', None) is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)
        # epoch = 5
    else:
        raise FileNotFoundError
    return epoch