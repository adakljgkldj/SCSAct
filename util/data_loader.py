import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
    })[config_type]

    train_loader, val_loader, lr_schedule, num_classes = None, None, [50, 75, 90], 0    # 类别数量被初始为0
    if args.in_dataset == "CIFAR-10":
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 100
    elif args.in_dataset == "imagenet":
        root = 'datasets/id_data/imagenet'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 1000
    elif args.in_dataset == "HAM10000":
        root = './HAM10000'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 5             # *********************************************************************************
    elif args.in_dataset == "lung":
        root = '../autodl-fs/DDCS/lung'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 4             # *********************************************************************************
    elif args.in_dataset == "cell":
        root = './cell'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 2             # *********************************************************************************
    elif args.in_dataset == "skin":
        root = './skin'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 2             # *********************************************************************************
    elif args.in_dataset == "blood":
        root = '../autodl-fs/DDCS/blood'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 4             # *********************************************************************************
    elif args.in_dataset == "NCT":
        root = '../autodl-fs/DDCS/NCT'
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 9
    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes,
    })
def get_loader_out(args, dataset=(''), config_type='default', split=('train', 'val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch_size
        },
    })[config_type]
    train_ood_loader, val_ood_loader = None, None
    if 'train' in split:
        if dataset[0].lower() == 'imagenet':
            train_ood_loader = torch.utils.data.DataLoader(
                ImageNet(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif dataset[0].lower() == 'tim':
            train_ood_loader = torch.utils.data.DataLoader(
                TinyImages(transform=config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch_size
        if val_dataset == 'SVHN':
            from util.svhn_loader import SVHN
            val_ood_loader = torch.utils.data.DataLoader(SVHN('datasets/ood_data/svhn/', split='test', transform=transform_test, download=False),
                                                       batch_size=batch_size, shuffle=False,
                                                        num_workers=2)
        elif val_dataset == 'dtd':
            transform = config.transform_test_largescale if args.in_dataset in {'imagenet'} else config.transform_test
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root="../autodl-fs/DDCS/datasets/ood_data/dtd", transform=transform),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'CIFAR-100':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='../autodl-fs/DDCS/data', train=False, download=True, transform=transform_test),
                                                       batch_size=batch_size, shuffle=True, num_workers=2)
        elif val_dataset == 'places50':
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("../autodl-fs/DDCS/datasets/ood_data/places50".format(val_dataset),
                                                          transform=config.transform_test_largescale), batch_size=batch_size, shuffle=False, num_workers=2)
        elif val_dataset == 'sun50':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("../autodl-fs/DDCS/datasets/ood_data/sun50".format(val_dataset),
                                                 transform=config.transform_test_largescale), batch_size=batch_size, shuffle=False,
                num_workers=2)
        elif val_dataset == 'inat':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder("../autodl-fs/DDCS/datasets/ood_data/inat".format(val_dataset),
                                                 transform=config.transform_test_largescale), batch_size=batch_size, shuffle=False,
                num_workers=2)
        elif val_dataset == 'imagenet':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join('../autodl-fs/DDCS/datasets/id_data/imagenet', 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'ISIC':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/ISIC_2019_Test', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'NCT':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/NCT-CRC-HE-100K', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'NCT2':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/NCT-CRC-HE-100K2', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'ham_near1':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/ham_near1', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'ham_near2':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/ham_near2', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'njcell':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/njcell', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'bnz':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/bnz', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'rx':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/rx', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'fallmud':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/fallmud', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'rxcs':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/rxcs', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'nkj':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/datasets/ood_data/nkj', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'lung_near':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('./datasets/ood_data/lung_near', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        elif val_dataset == 'blood':
            val_ood_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('../autodl-fs/DDCS/blood/val', config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        else:
            val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./datasets/ood_data/{}".format(val_dataset),
                                                          transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=2)

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
    })
