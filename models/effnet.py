from efficientnet_pytorch import EfficientNet
from torch import nn



def create_model(architecture, pretrained, freeze, num_classes):

    net = None

    # advprop: load pretrained weights
    if pretrained:
        net = EfficientNet.from_pretrained(architecture)
    else:
        net = EfficientNet.from_name(architecture)

    net._fc = nn.Linear(in_features=net._fc.in_features, out_features=num_classes, bias=True)

    if freeze and pretrained:
        print('Freezing parameters...')
        # freeze all parameters
        for param in net.parameters():
            param.requires_grad = False

        # unfreeze linear layer
        for p in net._fc.parameters():
            p.requires_grad  = True

    return net
