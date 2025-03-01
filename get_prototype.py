from __future__ import print_function
import argparse
import os
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import time
from util.metrics import compute_traditional_ood, compute_in
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_fun(args):
    def forward_threshold(inputs, model):
        if args.model_arch in {'mobilenet'}:
            logits = model.forward(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l)

        else:
            logits = model(inputs)
        return logits

    return forward_threshold


args = get_args()
forward_threshold = forward_fun(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def eval_ood_detector(args, mode_args):
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    method = args.method
    method_args = args.method_args
    name = args.name
    threshold_h = args.threshold_h
    threshold_l = args.threshold_l


    label_file = 'ID_prototype/lung.txt'              #

    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            image_name, label = line.strip().split()
            label_dict[image_name] = int(label)

    class_features = {}

    in_save_dir = os.path.join(base_dir, in_dataset, method, name)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split=('val'))
    testloaderIn, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes  # -
    method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)

    t0 = time.time()

    if True:

        ########################################In-distribution###########################################
        print("Processing in-distribution images")
        N = len(testloaderIn.dataset)
        count = 0
        for i, data in enumerate(testloaderIn):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            curr_batch_size = images.shape[0]
            inputs = images.float()

            features = model.forward_react(inputs, threshold_h)
            for j in range(features.shape[0]):
                image_filename, _ = testloaderIn.dataset.samples[i * curr_batch_size + j]
                label = label_dict[os.path.basename(image_filename)]

                if label not in class_features:
                    class_features[label] = []

                class_features[label].append(features[j].squeeze().detach().cpu().numpy())
        class_means = {}

        for label, features_list in class_features.items():
            class_means[label] = np.mean(features_list, axis=0)
            class_means_path = 'ID_prototype/mean.npy'
            print(len(class_means[label]))
            np.save(class_means_path, class_means)

    return


if __name__ == '__main__':
    args.method_args = dict()
    mode_args = dict()

    if args.method == "odin":
        args.method_args['temperature'] = 1000.0
        param_dict = {
            "CIFAR-10": {
                "resnet18": 0.01,
                "resnet18_cl1.0": 0.07,
            },
            "CIFAR-100": {
                "resnet18": 0.04,
                "resnet18_cl1.0": 0.04,
            },
            "imagenet": {
                "resnet50": 0.005,
                "resnet50_cl1.0": 0.0,
                "resnet18": 0.005,
                "mobilenet": 0.03,
                "mobilenet_cl1.3": 0.04,
            },
            "HAM10000": {
                "resnet50": 0.005,
                "resnet50_cl1.0": 0.0,
                "resnet18": 0.005,
                "mobilenet": 0.03,
                "mobilenet_cl1.3": 0.04,
            },
            "blood": {
                "resnet50": 0.005,
                "resnet50_cl1.0": 0.0,
                "resnet18": 0.005,
                "mobilenet": 0.03,
                "mobilenet_cl1.3": 0.04,
            }
        }
        args.method_args['magnitude'] = param_dict[args.in_dataset][args.name]
    if args.method == 'mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join('output/mahalanobis_hyperparams/', args.in_dataset, args.name, 'results.npy'),
            allow_pickle=True)
        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])
        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias
        args.method_args['sample_mean'] = sample_mean
        args.method_args['precision'] = precision
        args.method_args['magnitude'] = magnitude
        args.method_args['regressor'] = regressor
        args.method_args['num_output'] = 1
    print("eval ood detector")
    eval_ood_detector(args, mode_args)
