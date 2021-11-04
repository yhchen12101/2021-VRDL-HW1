from Modules import module
import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model related
    parser.add_argument("-m", "--model", default="cait_S36", type=str,
                        help="Model type.")
    parser.add_argument("--finetune", default=True,
                        help="finetune the pre-trained model.")
    parser.add_argument("--freeze", default=True,
                        help="freeze part of the pre-trained model.")
    parser.add_argument("--center_initialize", default=True,
                        help="using center of features to initial classifier weight.")
    parser.add_argument("--name", default=None,
                        help="Name of saved model.")

    # data related
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--mixup", default=True,
                        help="is using mixup?")
    parser.add_argument("--alpha", default=0.8, type=int,
                        help="alpha in mixup.")

    # training related
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='base learning rate for training')
    parser.add_argument('--decay', type=float, default=5e-5,
                        help='weight_decay for adam')
    parser.add_argument("-e", "--epochs", default=100, type=int,
                        help="Total training epoch.")
    parser.add_argument("--device", default=0, type=int,
                        help="GPU index to use, for cpu use -1.")
    parser.add_argument("-sc", "--scheduling", default=[35,65], nargs="*", type=int,
                        help="Epoch step where to reduce the learning rate.")
    parser.add_argument("-lr_decay", "--lr_decay", default=1/10, type=float,
                        help="LR multiplied by it.")
    
    
    args = parser.parse_args()
    args = vars(args) # Converting argparse Namespace to a dict.
    
    # build model
    model_type = args["model"]
    device = args["device"]

    if model_type.split("_")[0] == "resnet":
        model = module.ResNet(model_type, device = device)
    elif model_type.split("_")[0] == "cait":
        model = module.CaiT(model_type, device = device)
    
    model.to(device)
    model.initialize(center_initialize = args["center_initialize"])
    
    # train model
    model, losses, accs = utils.train(model, args)

    #save model
    if args["name"] is not None:
        name = args["name"]
    else:
        name = args["model"]
        if args["mixup"]:
            name += "_mixup_" 
        if args["finetune"]:
            name += "_finetune"

    torch.save(model.state_dict(), os.path.join("./save/" + name))

    # test
    test_pred = utils.test(model, device)



