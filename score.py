import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.mahalanobis_lib import get_Mahalanobis_score
from torch.autograd import Variable
from util.args_loader import get_args
import math

args = get_args()

def get_msp_score(inputs, model, forward_func, method_args, logits=None):
    if logits is None:
        with torch.no_grad():
            logits = forward_func(inputs, model)
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_msp_score_bats(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_bats(inputs)
            features = torch.where(features > feature_std, feature_std, features)  # ########
            features = torch.where(features < feature_mean, feature_mean, features)
            logits = model.forward_bats_head(features)

    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_msp_score_bats_react(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_react_bats(inputs, threshold=args.threshold_h)
            features = torch.where(features < (feature_std * lam + feature_mean), features,
                                   feature_std * lam + feature_mean)
            features = torch.where(features > (-feature_std * lam + feature_mean), features,
                                   -feature_std * lam + feature_mean)
            logits = model.forward_react_bats_head(features)

    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_msp_score_bats_lhact(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_lhact_feat(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l)
            features = torch.where(features < (feature_std * lam + feature_mean), features,
                                   feature_std * lam + feature_mean)
            features = torch.where(features > (-feature_std * lam + feature_mean), features,
                                   -feature_std * lam + feature_mean)
            logits = model.forward_lhact_bats_head(features)

    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_msp_score_bats_ddcs(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_ddcs_bats(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l,
                                               a=args.a, k=args.k)
            features = torch.where(features > feature_std, feature_std, features)
            features = torch.where(features < feature_mean, feature_mean, features)
            logits = model.forward_ddcs_bats_head(features)

    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_energy_score(inputs, model, forward_func, method_args, logits=None):
    if logits is None:
        with torch.no_grad():
            logits = forward_func(inputs, model)
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_energy_score_bats(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_bats(inputs)
            feature_mean.zero_()
            features = torch.where(features > feature_std, feature_std, features)
            features = torch.where(features < feature_mean, feature_mean, features)
            logits = model.forward_bats_head(features)
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_energy_score_bats_react(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_react_bats(inputs, threshold=args.threshold_h)
            features = torch.where(features < (feature_std * lam + feature_mean), features,
                                   feature_std * lam + feature_mean)
            features = torch.where(features > (-feature_std * lam + feature_mean), features,
                                   -feature_std * lam + feature_mean)
            logits = model.forward_react_bats_head(features)
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_energy_score_bats_lhact(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_lhact_feat(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l)
            features = torch.where(features < (feature_std * lam + feature_mean), features,
                                   feature_std * lam + feature_mean)
            features = torch.where(features > (-feature_std * lam + feature_mean), features,
                                   -feature_std * lam + feature_mean)
            logits = model.forward_lhact_bats_head(features)

    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_energy_score_bats_ddcs(inputs, model, feature_std, feature_mean, lam, logits=None):
    if logits is None:
        with torch.no_grad():
            features = model.forward_ddcs_bats(inputs, threshold_h=args.threshold_h, threshold_l=args.threshold_l,
                                               a=args.a, k=args.k)
            # features = torch.where(features < (feature_std * lam + feature_mean), features,feature_std * lam + feature_mean)
            # features = torch.where(features > (-feature_std * lam + feature_mean), features,-feature_std * lam + feature_mean)
            features = torch.where(features > feature_std, feature_std, features)
            features = torch.where(features < feature_mean, feature_mean, features)
            logits = model.forward_ddcs_bats_head(features)

    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_odin_score(inputs, model, forward_func, method_args):
    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad=True)
    outputs = forward_func(inputs, model)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    with torch.no_grad():
        outputs = forward_func(tempInputs, model)
    outputs = outputs / temper
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_odin_score_bats(inputs, model, feature_std, feature_mean, lam, method_args):
    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad=True)

    features = model.forward_bats(inputs)
    features = torch.where(features < (feature_std * lam + feature_mean), features, feature_std * lam + feature_mean)
    features = torch.where(features > (-feature_std * lam + feature_mean), features, -feature_std * lam + feature_mean)
    outputs = model.forward_bats_head(features)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)

    with torch.no_grad():

        features = model.forward_bats(tempInputs)
        features = torch.where(features < (feature_std * lam + feature_mean), features,
                               feature_std * lam + feature_mean)
        features = torch.where(features > (-feature_std * lam + feature_mean), features,
                               -feature_std * lam + feature_mean)
        outputs = model.forward_bats_head(features)
    outputs = outputs / temper
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']

    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output,
                                               magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores


def get_gradnorm_score(data_loader, model, temperature, num_classes, lam, feature_std, feature_mean, bats=False):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()

        features = model.forward_features(inputs)
        if bats:
            features = torch.where(features < (feature_std * lam + feature_mean), features,
                                   feature_std * lam + feature_mean)
            features = torch.where(features > (-feature_std * lam + feature_mean), features,
                                   -feature_std * lam + feature_mean)
        outputs = model.forward_head(features)

        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward(retain_graph=True)

        if num_classes == 1000:
            layer_grad = model.head.weight.grad.data
        else:
            layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def get_score(inputs, model, forward_func, method, method_args, logits=None):
    feature_std = torch.load("lung_resnet_max.pt").cuda()
    feature_mean = torch.load("lung_resnet_min.pt").cuda()

    lam = 2.03


    if method == "msp":
        scores = get_msp_score(inputs, model, forward_func, method_args, logits)
    elif method == "msp_bats":
        logits = None
        scores = get_msp_score_bats(inputs, model, feature_std, feature_mean, lam, logits)
    elif method == "msp_react_bats":
        logits = None
        scores = get_msp_score_bats_react(inputs, model, feature_std, feature_mean, lam, logits)
    elif method == "msp_lhact_bats":
        logits = None
        scores = get_msp_score_bats_lhact(inputs, model, feature_std, feature_mean, lam, logits)
    elif method == "msp_ddcs_bats":
        logits = None
        scores = get_msp_score_bats_ddcs(inputs, model, feature_std, feature_mean, lam, logits)

    elif method == "odin":
        scores = get_odin_score(inputs, model, forward_func, method_args)


    elif method == "energy":
        scores = get_energy_score(inputs, model, forward_func, method_args, logits)
    elif method == "energy_bats":
        logits = None
        scores = get_energy_score_bats(inputs, model, feature_std, feature_mean, lam, logits)
    elif method == "energy_react_bats":
        logits = None
        scores = get_energy_score_bats_react(inputs, model, feature_std, feature_mean, lam, logits)
    elif method == "energy_lhact_bats":
        logits = None
        scores = get_energy_score_bats_lhact(inputs, model, feature_std, feature_mean, lam, logits)
    elif method == "energy_ddcs_bats":
        logits = None
        scores = get_energy_score_bats_ddcs(inputs, model, feature_std, feature_mean, lam, logits)


    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    elif method == "gradnorm":
        score = get_gradnorm_score()
    return scores
