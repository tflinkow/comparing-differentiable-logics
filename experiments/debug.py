from __future__ import print_function

import torch
import torchvision

import copy

import sys
sys.path.append(".")
sys.path.append("..")

import matplotlib.pyplot as plt

def visualise_adv(model, inputs, adv, labels):
    model = copy.deepcopy(model)
    model.eval()

    with torch.no_grad():
        original_pred = torch.argmax(model(inputs[0].unsqueeze(0)), dim=1)
        adversarial_pred = torch.argmax(model(adv[0].unsqueeze(0)), dim=1)

    grid = torchvision.utils.make_grid([inputs[0], adv[0]], normalize=True).cpu()

    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f'True={labels[0]}, Pred(original)={original_pred.item()}, Pred(adv)={adversarial_pred.item()}')
    plt.show()

def print_pred_adv(model, inputs, adv, labels):
    get_probs = lambda m, x: torch.softmax(m(x), dim=1)[0].detach()
    display_vec = lambda x: ', '.join([f'{i}: {v:.2f}' for i, v in enumerate(x)])

    pred_inputs = get_probs(model, inputs)
    pred_adv = get_probs(model, adv)

    print('model(inputs):', f'[{display_vec(pred_inputs.cpu().numpy())}]')
    print('model(adv):', f'[{display_vec(pred_adv.cpu().numpy())}]')
    print('y:', labels[0].item())