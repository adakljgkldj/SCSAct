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
from score import get_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def forward_fun(args):
    def forward_threshold(inputs, model):
        if args.model_arch in {'mobilenet'}:
            logits = model.forward(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l, a=args.a,
                                   k=args.k)
        else:
            logits = model(inputs)
        return logits

    return forward_threshold

def forward_fun_forward(args):
    def forward_thr(inputs, model):
        logits = model.forward(inputs)
        return logits

    return forward_thr


def forward_fun_react(args):
    def forward_react(inputs, model):
        logits = model.forward_react(inputs, threshold=args.threshold_h)
        return logits

    return forward_react


def forward_fun_lhact(args):
    def forward_lhact(inputs, model):
        logits = model.forward_lhact(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l)
        return logits

    return forward_lhact

def forward_fun_ddcs(args):
    def forward_ddcs(inputs, model):
        logits = model.forward_ddcs(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l, a=args.a,
                                    k=args.k)
        return logits

    return forward_ddcs

def forward_resnet(args):
    def forward(inputs, model):
        logits = model.forward(inputs)
        return logits

    return forward
def forward_resnet_react(args):
    def forward(inputs, model):
        logits = model.forward_react(inputs, threshold=args.threshold_h)
        return logits

    return forward

def forward_resnet_lhact(args):
    def forward(inputs, model):
        logits = model.forward_lhact(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l)
        return logits

    return forward
def forward_resnet_ddcs(args):
    def forward(inputs, model):
        logits = model.forward_ddcs(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l, a=args.a,
                                    k=args.k)
        return logits

    return forward
def forward_mobilenet_ddcs(args):
    def forward_threshold(inputs, model):
        if args.model_arch in {'mobilenet'}:
            logits = model.forward(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l, a=args.a,
                                   k=args.k)
        else:
            logits = model(inputs)
        return logits

    return forward_threshold

def forward_mobilenet(args):
    def forward(inputs, model):
        logits = model.forward_thr(inputs)
        return logits

    return forward

def forward_mobilenet_react(args):
    def forward(inputs, model):
        logits = model._forward_impl_react(inputs, threshold=args.threshold_h)
        return logits

    return forward

def forward_mobilenet_lhact(args):
    def forward(inputs, model):
        logits = model.forward_lhact(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l)
        return logits

    return forward

args = get_args()
mobilenet_forward = forward_mobilenet(args)
mobilenet_react = forward_mobilenet_react(args)
mobilenet_lhact = forward_mobilenet_lhact(args)
mobilenet_ddcs = forward_mobilenet_ddcs(args)
resnet_forward = forward_resnet(args)
resnet_react = forward_resnet_react(args)
resnet_lhact = forward_resnet_lhact(args)
resnet_ddcs = forward_resnet_ddcs(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def eval_ood_detector(args, mode_args):
    start_time = time.time()
    base_dir = args.base_dir
    in_dataset = args.in_dataset
    out_datasets = args.out_datasets
    method = args.method
    method_args = args.method_args
    name = args.name
    a = args.a
    k = args.k

    in_save_dir = os.path.join(base_dir, in_dataset, args.model_arch, method,name)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_loader_in(args, split=('val'))
    testloaderIn, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    method_args['num_classes'] = num_classes
    model = get_model(args, num_classes, load_ckpt=True)

    t0 = time.time()
    if True:
        f1 = open(os.path.join(in_save_dir, "in_scores.txt"), 'w')
        g1 = open(os.path.join(in_save_dir, "in_labels.txt"), 'w')

        ########################################In-distribution###########################################
        print("Processing in-distribution images")
        N = len(testloaderIn.dataset)
        count = 0

        for j, data in enumerate(testloaderIn):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            curr_batch_size = images.shape[0]
            inputs = images.float()

            with torch.no_grad():
                logits = resnet_react(inputs, model)
                _, predicted = torch.max(logits.data, dim=1)
                print(labels, predicted)
                outputs = F.softmax(logits, dim=1)
                outputs = outputs.detach().cpu().numpy()
                preds = np.argmax(outputs, axis=1)
                confs = np.max(outputs, axis=1)

                for k in range(preds.shape[0]):
                    g1.write("{} {} {}\n".format(labels[k], preds[k], confs[k]))

            scores = get_score(inputs, model, resnet_react, method, method_args,logits=logits)

            for score in scores:
                f1.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
            t0 = time.time()

        f1.close()
        g1.close()

    # OOD evaluation
    for out_dataset in out_datasets:

        out_save_dir = os.path.join(in_save_dir,out_dataset)

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        f2 = open(os.path.join(out_save_dir, "out_scores.txt"), 'w')

        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        testloaderOut = get_loader_out(args, (None, out_dataset), split='val').val_ood_loader
        ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")

        N = len(testloaderOut.dataset)
        count = 0
        for j, data in enumerate(testloaderOut):

            images, labels = data
            images = images.cuda()
            curr_batch_size = images.shape[0]

            inputs = images.float()

            with torch.no_grad():

                logits = resnet_react(inputs, model)


            scores = get_score(inputs, model, resnet_react, method, method_args, logits=logits)
            for score in scores:
                f2.write("{}\n".format(score))

            count += curr_batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - t0))
            t0 = time.time()

        f2.close()

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"times: {execution_time} s")
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
            "lung": {
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
            },
            "NCT": {
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
    #
    eval_ood_detector(args, mode_args)
    compute_traditional_ood(args.base_dir, args.in_dataset, args.out_datasets, args.model_arch, args.method, args.name,
                            args.m, args.n)
    compute_in(args.base_dir, args.in_dataset, args.model_arch, args.method, args.name)

