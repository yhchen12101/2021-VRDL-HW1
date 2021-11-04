import os
from Modules import module
import utils
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=0, type=int,
                            help="GPU index to use, for cpu use -1.")
    parser.add_argument("-m", "--model", default="cait_S36", type=str,
                        help="Model type.")
    parser.add_argument('--path', help="model path.")

    args = parser.parse_args()
    args = vars(args) # Converting argparse Namespace to a dict.
    
    # build model
    model_type = args["model"]
    device = args["device"]

    if model_type.split("_")[0] == "resnet":
        model = module.ResNet(model_type, device = device)
    elif model_type.split("_")[0] == "cait":
        model = module.CaiT(model_type, device = device)

    #load model
    model.load_state_dict(torch.load(args["path"]))
    model.to(device)
    test_pred = utils.test(model, device)