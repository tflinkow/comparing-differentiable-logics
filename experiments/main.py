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

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data

from torchvision import datasets, transforms

from base.backends import TorchBackend
from base.logic import Logic
from base.dl2 import DL2
from base.fuzzy_logics import *

from experiments.constraints import *
from experiments.models import MnistNet, VGG16

from experiments.group_definitions import gtsrb_groups, cifar100_groups, cifar10_groups

from experiments.util import GradNorm, PGD, maybe
from experiments.debug import *

EpochInfo = namedtuple('EpochInfo', 'pred_acc constr_acc pred_loss constr_loss pred_loss_weight constr_loss_weight')

def train(model: torch.nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer, pgd: PGD, logic: Logic, constraint: Constraint, alpha: float, is_baseline: bool) -> EpochInfo:
    avg_pred_acc, avg_pred_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
    avg_constr_acc, avg_constr_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)

    model.train()

    with maybe(GradNorm(device, model), not is_baseline) as grad_norm:
        for batch_index, (data, target) in enumerate(train_loader, start=1):
            inputs, labels = data.to(device), target.to(device)

            outputs = model(inputs)
            ce_loss = F.cross_entropy(outputs, labels)

            correct = torch.mean(torch.argmax(outputs, dim=1).eq(labels).float())

            adv = pgd.attack(model, inputs, labels, logic, constraint, constraint.eps)

            with maybe(torch.no_grad(), is_baseline):
                dl_loss, sat = constraint.eval(model, inputs, adv, labels, logic, train=True)

            loss = ce_loss if is_baseline else grad_norm.weighted_loss(batch_index, ce_loss, dl_loss, alpha)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            avg_pred_acc += correct
            avg_pred_loss += ce_loss
            avg_constr_acc += sat
            avg_constr_loss += dl_loss

    return EpochInfo(
        pred_acc=avg_pred_acc.item() / float(batch_index),
        constr_acc=avg_constr_acc.item() / float(batch_index),
        pred_loss=avg_pred_loss.item() / float(batch_index),
        constr_loss=avg_constr_loss.item() / float(batch_index),
        pred_loss_weight=model.loss_weights[0].item(),
        constr_loss_weight=model.loss_weights[1].item()
    )

def test(model: torch.nn.Module, device: torch.device, test_loader: torch.utils.data.DataLoader, pgd: PGD, logic: Logic, constraint: Constraint) -> EpochInfo:
    test_ce_loss, test_dl_loss = torch.tensor(0., device=device), torch.tensor(0., device=device)
    correct, constr = torch.tensor(0., device=device), torch.tensor(0., device=device)

    model.eval()

    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader, start=1):
            inputs, labels = data.to(device), target.to(device)

            outputs = model(inputs)
            ce_loss = F.cross_entropy(outputs, labels)

            adv = pgd.attack(model, inputs, labels, logic, constraint, constraint.eps)

            dl_loss, sat = constraint.eval(model, inputs, adv, labels, logic, train=False)

            test_ce_loss += ce_loss
            test_dl_loss += dl_loss

            pred = outputs.max(dim=1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum()

            constr += sat

    return EpochInfo(
        pred_acc=correct.item() / len(test_loader.dataset), 
        constr_acc=constr.item() / len(test_loader.dataset),
        pred_loss=test_ce_loss.item() / len(test_loader.dataset),
        constr_loss=test_dl_loss.item() / len(test_loader.dataset),
        pred_loss_weight=None,
        constr_loss_weight=None
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
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-set', type=str, required=True, choices=['mnist', 'gtsrb', 'cifar10', 'cifar100'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pgd-steps', type=int, default=10)
    parser.add_argument('--pgd-gamma', type=float, default=float(48/255))
    parser.add_argument('--logic', type=str, default=None, choices=[l.abbrv for l in logics])
    parser.add_argument('--constraint', type=str, required=True, help='one of Robustness(eps: float, delta: float), Groups(eps: float, delta: float), or ClassSimilarity(eps: float)')
    parser.add_argument('--reports-dir', type=str, default='../reports')
    parser.add_argument('--grad-norm-alpha', type=float, default=0.1)
    args = parser.parse_args()

    kwargs = { 'batch_size': args.batch_size }

    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')

        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        kwargs.update({ 'num_workers': 4, 'pin_memory': True })
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if args.logic == None:
        logic = logics[0] # need some logic loss for PGD even for baseline
        baseline = True
    else:
        logic = next(l for l in logics if l.abbrv == args.logic)
        baseline = False

    def Robustness(eps: float, delta: float) -> RobustnessConstraint:
        return RobustnessConstraint(torch.tensor(eps, device=device), torch.tensor(delta, device=device))

    def Groups(eps: float, delta: float) -> GroupConstraint:
        if args.data_set == 'gtsrb':
            groups = gtsrb_groups
        elif args.data_set == 'cifar100':
            groups = cifar100_groups
        else:
            assert False, 'Group constraint is designed for GTSRB and CIFAR100 only'

        return GroupConstraint(torch.tensor(eps, device=device), torch.tensor(delta, device=device), groups)
    
    def ClassSimilarity(eps: float) -> ClassSimilarityConstraint:
        assert args.data_set == 'cifar10', 'ClassSimilarity constraint is designed for CIFAR10 only'
        return ClassSimilarityConstraint(torch.tensor(eps, device=device), cifar10_groups)

    constraint: Constraint = eval(args.constraint)

    if args.data_set == 'mnist':
        mean, std = (.1307,), (.3081,)

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
    elif args.data_set == 'gtsrb':
        mean, std = (.3403, .3121, .3214), (.2724, .2608, .2669)

        transform_train = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.GTSRB('../data', split="train", download=True, transform=transform_train)
        dataset_test = datasets.GTSRB('../data', split="test", download=True, transform=transform_test)

        model = VGG16(43).to(device)
    elif args.data_set == 'cifar10':
        mean, std = (.4914, .4822, .4465), (.2023, .1994, .2010)
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data', train=False, download=True, transform=transform_test)

        model = VGG16(10).to(device)
    elif args.data_set == 'cifar100':
        mean, std = (.5071, .4867, .4408), (.2675, .2565, .2761)

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        dataset_train = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR100('../data', train=False, download=True, transform=transform_test)

        model = VGG16(100).to(device)

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, **kwargs)

    pgd = PGD(device, args.pgd_steps, mean, std, args.pgd_gamma)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.reports_dir, exist_ok=True)

    if isinstance(constraint, RobustnessConstraint):
        folder = 'robustness'
    elif isinstance(constraint, GroupConstraint):
        folder = 'groups'
    elif isinstance(constraint, ClassSimilarityConstraint):
        folder = 'class-similarity'

    file_name = f'{args.reports_dir}/{folder}/{args.data_set}/{logic.name if not baseline else "Baseline"}.csv'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        csvfile.write(f'#{sys.argv}\n')
        writer.writerow(['Epoch', 'Train-P-Loss', 'Train-C-Loss', 'Train-P-Loss-Weight', 'Train-C-Loss-Weight', 'Train-P-Acc', 'Train-C-Acc', 'Test-P-Loss', 'Test-C-Loss', 'Test-P-Acc', 'Test-C-Acc', 'Time'])

        for epoch in range(0, args.epochs + 1):
            start = time.time()

            if epoch > 0:
                train_info = train(model, device, train_loader, optimizer, pgd, logic, constraint, args.grad_norm_alpha, is_baseline=baseline)
                train_time = time.time() - start

                print(f'Epoch {epoch}/{args.epochs}\t {args.constraint} on {args.data_set}, {logic.name if not baseline else "Baseline"} \t TRAIN \t P-Acc: {train_info.pred_acc:.4f}\t C-Acc: {train_info.constr_acc:.4f}\t CE-Loss: {train_info.pred_loss:.2f}\t DL-Loss: {train_info.constr_loss:.2f}\t Time (Train) [s]: {train_time:.1f}')
            else:
                train_info = EpochInfo(0., 0., 0., 0., 1., 1.)
                train_time = 0.

            test_info = test(model, device, test_loader, pgd, logic, constraint)
            test_time = time.time() - start - train_time

            writer.writerow([epoch, \
                             train_info.pred_loss, train_info.constr_loss, train_info.pred_loss_weight, train_info.constr_loss_weight, train_info.pred_acc, train_info.constr_acc, \
                             test_info.pred_loss, test_info.constr_loss, test_info.pred_acc, test_info.constr_acc, \
                             train_time])

            print(f'Epoch {epoch}/{args.epochs}\t {args.constraint} on {args.data_set}, {logic.name if not baseline else "Baseline"} \t TEST \t P-Acc: {test_info.pred_acc:.4f}\t C-Acc: {test_info.constr_acc:.4f}\t CE-Loss: {test_info.pred_loss:.2f}\t DL-Loss: {test_info.constr_loss:.2f}\t Time (Test) [s]: {test_time:.1f}')
            print(f'===')

if __name__ == '__main__':
    main()