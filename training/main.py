from __future__ import print_function

from collections import namedtuple

import argparse

import time
import os
import csv

import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np

import onnx

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data

from torchvision import datasets, transforms
from torchvision.utils import save_image

from base.backends import TorchBackend
from base.logic import Logic
from base.dl2 import DL2
from base.fuzzy_logics import *

from training.constraints import *
from training.models import *

from training.group_definitions import gtsrb_groups, cifar10_groups

from training.util import *
from training.grad_norm import *
from training.attacks import *

EpochInfoTrain = namedtuple('EpochInfoTrain', 'pred_acc constr_acc constr_sec pred_loss random_loss constr_loss pred_loss_weight constr_loss_weight input_img adv_img random_img')
EpochInfoTest = namedtuple('EpochInfoTest', 'pred_acc constr_acc constr_sec pred_loss random_loss constr_loss input_img adv_img random_img vacuously_true')

def train(model: torch.nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer, oracle: Attack, grad_norm: GradNorm, logic: Logic, constraint: Constraint, with_dl: bool) -> EpochInfoTrain:
    avg_pred_acc, avg_pred_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
    avg_constr_acc, avg_constr_sec, avg_constr_loss, avg_random_loss = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)

    images = { 'input': None, 'random': None, 'adv': None}

    model.train()

    for _, (data, target) in enumerate(train_loader, start=1):
        inputs, labels = data.to(device), target.to(device)

        # forward pass for prediction accuracy
        outputs = model(inputs)
        ce_loss = F.cross_entropy(outputs, labels)
        correct = torch.mean(torch.argmax(outputs, dim=1).eq(labels).float())

        # get random + adversarial samples
        with torch.no_grad():
            random = oracle.uniform_random_sample(inputs)

        adv = oracle.attack(model, inputs, labels, logic, constraint)

        # forward pass for constraint accuracy (constraint satisfaction on random samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(model, inputs, random, labels, logic, reduction='mean')

        # forward pass for constraint security (constraint satisfaction on adversarial samples)
        with maybe(torch.no_grad(), not with_dl):
            loss_adv, sat_adv = constraint.eval(model, inputs, adv, labels, logic, reduction='mean')

        optimizer.zero_grad(set_to_none=True)

        if not with_dl:
            ce_loss.backward()
            optimizer.step()
        else:
            grad_norm.balance(ce_loss, loss_adv)

        avg_pred_acc += correct
        avg_pred_loss += ce_loss
        avg_constr_acc += sat_random
        avg_constr_sec += sat_adv
        avg_constr_loss += loss_adv
        avg_random_loss += loss_random

        # save one original image, random sample, and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, inputs.size(0) - 1)
        images['input'], images['random'], images['adv'] = inputs[i], random[i], adv[i]

    if with_dl:
        grad_norm.renormalise()

    return EpochInfoTrain(
        pred_acc=avg_pred_acc.item() / len(train_loader),
        constr_acc=avg_constr_acc.item() / len(train_loader),
        constr_sec=avg_constr_sec.item() / len(train_loader),
        pred_loss=avg_pred_loss.item() / len(train_loader),
        random_loss=avg_random_loss.item() / len(train_loader),
        constr_loss=avg_constr_loss.item() / len(train_loader),
        pred_loss_weight=grad_norm.weights[0].item(),
        constr_loss_weight=grad_norm.weights[1].item(),
        input_img=images['input'],
        adv_img=images['adv'],
        random_img=images['random']
    )

def test(model: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader, oracle: Attack, logic: Logic, constraint: Constraint) -> EpochInfoTest:
    correct, constr_acc, constr_sec = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)
    avg_pred_loss, avg_constr_loss, avg_random_loss = torch.tensor(0., device=device), torch.tensor(0., device=device), torch.tensor(0., device=device)

    record_vacuously_true = isinstance(constraint, EvenOddConstraint) or isinstance(constraint, ClassSimilarityConstraint)

    if record_vacuously_true:
        vacuously_true = torch.zeros(2 if isinstance(constraint, EvenOddConstraint) else 10, device=device)

    total_samples = 0

    images = { 'input': None, 'random': None, 'adv': None}

    model.eval()

    for _, (data, target) in enumerate(test_loader, start=1):
        inputs, labels = data.to(device), target.to(device)
        total_samples += inputs.size(0)

        with torch.no_grad():
            # forward pass for prediction accuracy
            outputs = model(inputs)
            avg_pred_loss += F.cross_entropy(outputs, labels, reduction='sum')
            pred = outputs.max(dim=1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum()

            # get random samples (no grad)
            random = oracle.uniform_random_sample(inputs)

        # get adversarial samples (requires grad)
        adv = oracle.attack(model, inputs, labels, logic, constraint)

        # forward passes for constraint accuracy (constraint satisfaction on random samples) + constraint security (constraint satisfaction on adversarial samples)
        with torch.no_grad():
            loss_random, sat_random = constraint.eval(model, inputs, random, labels, logic, reduction='sum')
            loss_adv, sat_adv = constraint.eval(model, inputs, adv, labels, logic, reduction='sum')

            if record_vacuously_true:
                vacuously_true += constraint.get_vacuously_true(model, adv)

            constr_acc += sat_random
            constr_sec += sat_adv

            avg_random_loss += loss_random
            avg_constr_loss += loss_adv

        # save one original image, random sample, and adversarial sample image (for debugging, inspecting attacks)
        i = np.random.randint(0, inputs.size(0) - 1)
        images['input'], images['random'], images['adv'] = inputs[i], random[i], adv[i]

    return EpochInfoTest(
        pred_acc=correct.item() / total_samples, 
        constr_acc=constr_acc.item() / total_samples,
        constr_sec=constr_sec.item() / total_samples,
        pred_loss=avg_pred_loss.item() / total_samples,
        random_loss=avg_random_loss.item() / total_samples,
        constr_loss=avg_constr_loss.item() / total_samples,
        input_img=images['input'],
        adv_img=images['adv'],
        random_img=images['random'],
        vacuously_true=(vacuously_true / total_samples) if record_vacuously_true else -1.
    )

def main():
    backend = TorchBackend()

    logics: list[Logic] = [
        DL2(backend),
        GoedelFuzzyLogic(backend),
        KleeneDienesFuzzyLogic(backend),
        LukasiewiczFuzzyLogic(backend),
        ReichenbachFuzzyLogic(backend),
        GoguenFuzzyLogic(backend),
        ReichenbachSigmoidalFuzzyLogic(backend),
        YagerFuzzyLogic(backend)
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--data-set', type=str, required=True, choices=['mnist', 'fmnist', 'gtsrb', 'cifar10'])
    parser.add_argument('--constraint', type=str, required=True, help='one of StandardRobustness(eps: float, delta: float), StrongClassificationRobustness(eps: float, delta: float), Groups(eps: float, delta: float), EvenOdd(eps: float, delta: float, gamma: float), or ClassSimilarity(eps: float, delta: float)')
    parser.add_argument('--oracle', type=str, default='apgd', choices=['apgd', 'pgd'], help='AutoPGD (apgd) or vanilla PGD')
    parser.add_argument('--oracle-steps', type=int, default=20, help='number of oracle iterations')
    parser.add_argument('--oracle-restarts', type=int, default=10, help='number of oracle random restarts')
    parser.add_argument('--pgd-gamma', type=float, default=100, help='PGD step_size = eps / gamma (only for vanilla PGD)')
    parser.add_argument('--delay', type=int, default=0, help='number of epochs to wait before introducing constraint loss')
    parser.add_argument('--logic', type=str, default=None, choices=[l.abbrv for l in logics], help='the differentiable logic to use for training with the constraint, or None')
    parser.add_argument('--results-dir', type=str, default='../results', help='directory in which to save .onnx and .csv files')
    parser.add_argument('--initial-dl-weight', type=float, default=1.)
    parser.add_argument('--grad-norm-alpha', type=float, default=.12, help='restoring force for GradNorm')
    parser.add_argument('--grad-norm-lr', type=float, default=None, help='learning rate for GradNorm weights, equal to --lr if not specified')
    parser.add_argument('--save-onnx', action='store_true', help='save .onnx file after training')
    parser.add_argument('--save-imgs', action='store_true', help='save one input image, random image, and adversarial image per epoch')
    args = parser.parse_args()

    kwargs = { 'batch_size': args.batch_size }

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')

        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        kwargs.update({ 'num_workers': 4, 'pin_memory': True })
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.logic == None:
        logic = logics[0] # need some logic loss for oracle even for baseline
        is_baseline = True
    else:
        logic = next(l for l in logics if l.abbrv == args.logic)
        is_baseline = False

    def StandardRobustness(eps: float, delta: float) -> StandardRobustnessConstraint:
        return StandardRobustnessConstraint(device, eps, delta)

    def EvenOdd(eps: float, delta: float, gamma: float) -> EvenOddConstraint:
        return EvenOddConstraint(device, eps, delta, gamma)

    def StrongClassificationRobustness(eps: float, delta: float) -> StrongClassificationRobustnessConstraint:
        return StrongClassificationRobustnessConstraint(device, eps, delta)

    def Groups(eps: float, delta: float) -> GroupConstraint:
        assert args.data_set == 'gtsrb', 'Group constraint is designed for GTSRB only'
        return GroupConstraint(device, eps, delta, gtsrb_groups)

    def ClassSimilarity(eps: float, delta: float) -> ClassSimilarityConstraint:
        assert args.data_set == 'cifar10', 'ClassSimilarity constraint is designed for CIFAR10'
        return ClassSimilarityConstraint(device, eps, cifar10_groups, delta)

    constraint: Constraint = eval(args.constraint)
    print(f'constraint.eps={constraint.eps}')

    if args.data_set == 'mnist':
        mean, std = (.1307,), (.3081,)

        if isinstance(constraint, StrongClassificationRobustnessConstraint):
            mean, std = (0.), (1.) # only used for the verification experiment

        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.MNIST('../data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.MNIST('../data', train=False, download=True, transform=transform_test)

        model = MnistNet().to(device)
    elif args.data_set == 'fmnist':
        mean, std = (.5,), (.5,)

        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.FashionMNIST('../data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.FashionMNIST('../data', train=False, download=True, transform=transform_test)

        model = MnistNet().to(device)
    elif args.data_set == 'gtsrb':
        mean, std = (.3403, .3121, .3214), (.2724, .2608, .2669)

        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=5, translate=(.1, .1)),
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.GTSRB('../data', split="train", download=True, transform=transform_train)
        dataset_test = datasets.GTSRB('../data', split="test", download=True, transform=transform_test)

        model = GTSRBNet().to(device)
    elif args.data_set == 'cifar10':
        mean, std = (.4914, .4822, .4465), (.2023, .1994, .2010)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data', train=False, download=True, transform=transform_test)

        model = Cifar10Net().to(device)

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, drop_last=True, **kwargs)

    print(f'len(dataset_train)={len(dataset_train)} len(dataset_test)={len(dataset_test)}')
    print(f'len(train_loader)={len(train_loader)} len(test_loader)={len(test_loader)}')

    if isinstance(constraint, StrongClassificationRobustnessConstraint):
        model = MnistNetSmall().to(device) # only used for the verification experiment

    if args.oracle == 'pgd':
        oracle = PGD(device, args.oracle_steps, args.oracle_restarts, mean, std, constraint.eps, args.pgd_gamma)
        oracle_test = PGD(device, args.oracle_steps * 2, args.oracle_restarts, mean, std, constraint.eps, args.pgd_gamma)
    elif args.oracle == 'apgd':
        oracle = APGD(device, args.oracle_steps, args.oracle_restarts, mean, std, constraint.eps)
        oracle_test = APGD(device, args.oracle_steps * 2, args.oracle_restarts, mean, std, constraint.eps)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    grad_norm = GradNorm(model, device, optimizer, lr=args.grad_norm_lr if args.grad_norm_lr is not None else args.lr, alpha=args.grad_norm_alpha, initial_dl_weight=args.initial_dl_weight)

    if isinstance(constraint, StandardRobustnessConstraint):
        folder = 'standard-robustness'
    elif isinstance(constraint, StrongClassificationRobustnessConstraint):
        folder = 'strong-classification-robustness'
    elif isinstance(constraint, EvenOddConstraint):
        folder = 'even-odd'
    elif isinstance(constraint, GroupConstraint):
        folder = 'groups'
    elif isinstance(constraint, ClassSimilarityConstraint):
        folder = 'class-similarity'
    else:
        assert False, f'unknown constraint {constraint}!'

    folder_name = f'{args.results_dir}/{folder}/{args.data_set}'
    file_name = f'{folder_name}/{logic.name if not is_baseline else "Baseline"}'

    report_file_name = f'{file_name}.csv'
    model_file_name = f'{file_name}.onnx'

    os.makedirs(folder_name, exist_ok=True)

    if args.save_imgs:
        save_dir = f'../saved_imgs/{folder}/{args.data_set}/{logic.name if not is_baseline else "Baseline"}'
        os.makedirs(save_dir, exist_ok=True)

    def save_imgs(info: EpochInfoTrain | EpochInfoTest, epoch):
        if not args.save_imgs:
            return

        def save_img(img: torch.Tensor, name: str):
            save_image(oracle.denormalise(img), os.path.join(save_dir, name))

        if isinstance(info, EpochInfoTrain):
            prefix = 'train'
        else:
            prefix = 'test'

        save_img(info.input_img, f'{epoch}-{prefix}_input.png')
        save_img(info.adv_img, f'{epoch}-{prefix}_adv.png')
        save_img(info.random_img, f'{epoch}-{prefix}_random.png')

    print(f'using device {device}')
    print(f'#model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # for constraints involving implication, record percentage of constraints being vacuously true
    with_extra_info = isinstance(constraint, EvenOddConstraint) or isinstance(constraint, ClassSimilarityConstraint)

    if isinstance(constraint, EvenOddConstraint):
        extra_columns = 2
    elif isinstance(constraint, ClassSimilarityConstraint):
        extra_columns = 10
    else:
        extra_columns = 0

    with open(report_file_name, 'w', buffering=1, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        csvfile.write(f'#{sys.argv}\n')
        writer.writerow(['Epoch', 'Train-P-Loss', 'Train-R-Loss', 'Train-C-Loss', 'Train-P-Loss-Weight', 'Train-C-Loss-Weight', 'Train-P-Acc', 'Train-C-Acc', 'Train-C-Sec', 'Test-P-Acc', 'Test-C-Acc', 'Test-C-Sec', 'Train-Time', 'Test-Time'] + [f'Extra{i}' for i in range(extra_columns)])

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                with_dl = (epoch > args.delay) and (not is_baseline)
                train_info = train(model, device, train_loader, optimizer, oracle, grad_norm, logic, constraint, with_dl)
                train_time = time.time() - start

                save_imgs(train_info, epoch)

                print(f'Epoch {epoch}/{args.epochs}\t {args.constraint} on {args.data_set}, {logic.name if not is_baseline else "Baseline"} \t TRAIN \t P-Acc: {train_info.pred_acc:.2f} \t C-Acc: {train_info.constr_acc:.2f}\t C-Sec: {train_info.constr_sec:.2f}\t P-Loss: {train_info.pred_loss:.2f}\t R-Loss: {train_info.random_loss:.2f}\t DL-Loss: {train_info.constr_loss:.2f}\t Time (Train) [s]: {train_time:.1f}')
            else:
                train_info = EpochInfoTrain(0., 0., 0., 0., 0., 0., 1., 1., None, None, None)
                train_time = 0.

            test_info = test(model, device, test_loader, oracle_test, logic, constraint)
            test_time = time.time() - start - train_time

            save_imgs(test_info, epoch)

            writer.writerow([epoch, \
                             train_info.pred_loss, train_info.random_loss, train_info.constr_loss, train_info.pred_loss_weight, train_info.constr_loss_weight, train_info.pred_acc, train_info.constr_acc, train_info.constr_sec, \
                             test_info.pred_acc, test_info.constr_acc, test_info.constr_sec, \
                             train_time, test_time] \
                            + ([v.item() for v in test_info.vacuously_true] if with_extra_info else []))

            if with_extra_info:
                print(f'impl vacuously true=[{" ".join([f"{x:.2f}" for x in test_info.vacuously_true])}]')

            print(f'Epoch {epoch}/{args.epochs}\t {args.constraint} on {args.data_set}, {logic.name if not is_baseline else "Baseline"} \t TEST \t P-Acc: {test_info.pred_acc:.2f}\t C-Acc: {test_info.constr_acc:.2f}\t C-Sec: {test_info.constr_sec:.2f}\t P-Loss: {test_info.pred_loss:.2f}\t R-Loss: {test_info.random_loss:.2f}\t DL-Loss: {test_info.constr_loss:.2f}\t Time (Test) [s]: {test_time:.1f}')
            print(f'===')

    if args.save_onnx:
        inputs, _ = next(iter(train_loader))
        _, c, h, w = inputs.shape

        torch.onnx.export(
            model.eval(),
            torch.randn(args.batch_size, c, h, w, requires_grad=True).to(device=device),
            model_file_name,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': { 0: 'batch_size' }, 'output': { 0: 'batch_size' }},
        )

        onnx_model = onnx.load(model_file_name)
        onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    main()