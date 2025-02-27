import os
import torch
def get_model(args, num_classes, load_ckpt=True):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "./Imagenet_resnet34.pth"
            model = resnet34(num_classes=num_classes)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes)
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    elif args.in_dataset == 'HAM10000':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "./MobileNetV3-master/weights/4resnet80.pth"
            model = resnet34(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            weight_path = "./MobileNetV3-master/weights/3nrescnn90.pth"
            model = mobilenet_v2(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'resnet_cnn':
            from models.resnet_with_transformer import resnet34
            model = resnet34(num_classes=num_classes, pretrained=True)
    elif args.in_dataset == 'lung':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "../autodl-fs/DDCS/lung_weight4/resnet30.pth"  #
            model = resnet34(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            weight_path = "../autodl-fs/DDCS/lung_mob_weight/resnet20.pth"
            model = mobilenet_v2(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'resnet_cnn':
            from models.resnet_with_transformer import resnet34
            model = resnet34(num_classes=num_classes, pretrained=True)
    elif args.in_dataset == 'cell':
        if args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "./cell_weight/resnet40.pth"
            model = resnet34(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
    elif args.in_dataset == 'skin':
        if args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "./skin_weight/resnet30.pth"
            model = resnet34(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
    elif args.in_dataset == 'blood':
        if args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "../autodl-fs/DDCS/blood_weight3/resnet10.pth"
            model = resnet34(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            weight_path = "../autodl-fs/DDCS/blood_mob_weight/resnet20.pth"
            model = mobilenet_v2(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
    elif args.in_dataset == 'NCT':
        if args.model_arch == 'resnet':
            from models.resnet import resnet34
            weight_path = "../autodl-fs/DDCS/NCT_weight/resnet30.pth"
            model = resnet34(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
        elif args.model_arch == 'mobilenet':
            from models.mobilenet import mobilenet_v2
            weight_path = "../autodl-fs/DDCS/blood_mob_weight/resnet20.pth"
            model = mobilenet_v2(num_classes=num_classes, pretrained=False)
            model.load_state_dict(torch.load(weight_path))
    else:
        if args.model_arch == 'resnet50':
            from models.resnet import resnet50_cifar
            model = resnet50_cifar(num_classes=num_classes, method=args.method)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model
