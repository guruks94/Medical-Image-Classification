import os

import torch
import torch.nn as nn
import torchvision.models as models


def ClassificationNet(arch_name, num_class, conv=None, weight=None, activation=None):
    if weight is None:
        weight = "none"

    model = models.__dict__[arch_name](pretrained=False)

    if arch_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        if activation is None:
            model.fc = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.fc = nn.Sequential(
                nn.Linear(kernelCount, num_class), nn.Sigmoid())

        # init the fc layer
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        kernelCount = model.classifier.in_features
        if activation is None:
            model.classifier = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.classifier = nn.Sequential(
                nn.Linear(kernelCount, num_class), nn.Sigmoid())

        # init the classifier layer
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[0].bias.data.zero_()
    elif arch_name.lower().startswith("efficientnet"):
        kernelCount = model.classifier[1].in_features
        if activation is None:
            model.classifier = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.classifier = nn.Sequential(
                nn.Linear(kernelCount, num_class), nn.Sigmoid())

        # init the classifier layer
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[0].bias.data.zero_()

    def _weight_loading_check(_arch_name, _activation, _msg):
        if len(_msg.missing_keys) != 0:
            if _arch_name.lower().startswith("resnet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
                else:
                    assert set(_msg.missing_keys) == {
                        "fc.0.weight", "fc.0.bias"}
            elif _arch_name.lower().startswith("densenet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {
                        "classifier.weight", "classifier.bias"}
                else:
                    assert set(_msg.missing_keys) == {
                        "classifier.0.weight", "classifier.0.bias"}

    if weight.lower() == "imagenet":
        pretrained_model = models.__dict__[arch_name](pretrained=True)
        state_dict = pretrained_model.state_dict()

        # delete fc layer
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded supervised ImageNet pre-trained model")
    elif os.path.isfile(weight):
        checkpoint = torch.load(weight, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        state_dict = {k.replace("module.encoder_q.", "")
                                : v for k, v in state_dict.items()}

        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded pre-trained model '{}'".format(weight))
        print("missing keys:", msg.missing_keys)

    # reinitialize fc layer again
    if arch_name.lower().startswith("resnet"):
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
    elif arch_name.lower().startswith("efficientnet"):
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)

    return model


def build_classification_model(args):
    if args.init.lower() == "random" or args.init.lower() == "imagenet":
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=args.init,
                                  activation=args.activate)

    else:
        model = ClassificationNet(args.model_name.lower(), args.num_class, weight=args.proxy_dir,
                                  activation=args.activate)

    return model


def save_checkpoint(state, filename='model'):

    torch.save(state, filename + '.pth.tar')
