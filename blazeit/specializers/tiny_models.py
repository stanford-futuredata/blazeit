from .trn import create_trn
from .vgg import create_tvgg, create_noscope

def create_tiny_model(tiny_name, nb_classes, weights=None):
    if tiny_name.startswith('trn'):
        weights = 'imagenet' if weights is None else weights
        return create_trn(tiny_name, nb_classes, weights=weights)
    elif tiny_name.startswith('tvgg'):
        weights = 'random' if weights is None else weights
        return create_tvgg(tiny_name, nb_classes, weights=weights)
    elif tiny_name.startswith('ns'):
        return create_noscope(tiny_name, nb_classes)
    else:
        raise NotImplementedError
