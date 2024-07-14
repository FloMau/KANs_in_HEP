import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mplhep as hep
from kan import KAN


def plot_custom(model, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None, tick_size=20):
    '''
    plot KAN with custom tick size for in_vars
    
    Args:
    -----
        folder : str
            the folder to store pngs
        beta : float
            positive number. control the transparency of each activation. transparency = tanh(beta*l1).
        mask : bool
            If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
        mode : bool
            "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
        scale : float
            control the size of the diagram
        in_vars: None or list of str
            the name(s) of input variables
        out_vars: None or list of str
            the name(s) of output variables
        title: None or str
            title
        tick_size: int
            size of the tick labels
        
    Returns:
    --------
        Figure
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    depth = len(model.width) - 1
    for l in range(depth):
        w_large = 2.0
        for i in range(model.width[l]):
            for j in range(model.width[l + 1]):
                rank = torch.argsort(model.acts[l][:, i])
                fig, ax = plt.subplots(figsize=(w_large, w_large))

                num = rank.shape[0]

                symbol_mask = model.symbolic_fun[l].mask[j][i]
                numerical_mask = model.act_fun[l].mask.reshape(model.width[l + 1], model.width[l])[j][i]
                if symbol_mask > 0. and numerical_mask > 0.:
                    color = 'purple'
                    alpha_mask = 1
                if symbol_mask > 0. and numerical_mask == 0.:
                    color = "red"
                    alpha_mask = 1
                if symbol_mask == 0. and numerical_mask > 0.:
                    color = "black"
                    alpha_mask = 1
                if symbol_mask == 0. and numerical_mask == 0.:
                    color = "white"
                    alpha_mask = 0

                # if tick == True:
                #     ax.tick_params(axis="y", direction="in", pad=0, labelsize=0.3*tick_size)
                #     ax.tick_params(axis="x", direction="in", pad=-15, labelsize=0.3*tick_size)
                #     x_min, x_max, y_min, y_max = model.get_range(l, i, j, verbose=False)
                #     # print("GET RANGE", model.get_range(l, i, j, verbose=False))
                #     plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                #     plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                # else:
                #     plt.xticks([])
                #     plt.yticks([])
                ax.tick_params(axis="y", direction="in", pad=0, labelsize=10)
                ax.tick_params(axis="x", direction="in", pad=0, labelsize=10)
                if alpha_mask == 1:
                    plt.gca().patch.set_edgecolor('black')
                else:
                    plt.gca().patch.set_edgecolor('white')
                plt.gca().patch.set_linewidth(1.5)

                plt.plot(model.acts[l][:, i][rank].cpu().detach().numpy(), model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                if sample == True:
                    plt.scatter(model.acts[l][:, i][rank].cpu().detach().numpy(), model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                plt.gca().spines[:].set_color(color)

                lock_id = model.act_fun[l].lock_id[j * model.width[l] + i].long().item()
                if lock_id > 0:
                    im = plt.imread(f'{folder}/lock.png')
                    newax = fig.add_axes([0.15, 0.7, 0.15, 0.15])
                    plt.text(500, 400, lock_id, fontsize=15)
                    newax.imshow(im)
                    newax.axis('off')
                plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=100)
                plt.close()

    def score2alpha(score):
        return np.tanh(beta * score)

    if mode == "supervised":
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in model.acts_scale]
    elif mode == "unsupervised":
        alpha = [score2alpha(score.cpu().detach().numpy()) for score in model.acts_scale_std]

    # draw skeleton
    width = np.array(model.width)
    A = 1
    y0 = 0.4  # 0.4

    neuron_depth = len(width)
    min_spacing = A / np.maximum(np.max(width), 5)

    max_neuron = np.max(width)
    max_num_weights = np.max(width[:-1] * width[1:])
    y1 = 0.4 / np.maximum(max_num_weights, 3)

    fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * y0))

    # plot scatters and lines
    for l in range(neuron_depth):
        n = width[l]
        spacing = A / n
        for i in range(n):
            plt.scatter(1 / (2 * n) + i / n, l * y0, s=min_spacing ** 2 * 10000 * scale ** 2, color='black')

            if l < neuron_depth - 1:
                n_next = width[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j

                    symbol_mask = model.symbolic_fun[l].mask[j][i]
                    numerical_mask = model.act_fun[l].mask.reshape(model.width[l + 1], model.width[l])[j][i]
                    if symbol_mask == 1. and numerical_mask == 1.:
                        color = 'purple'
                        alpha_mask = 1.
                    if symbol_mask == 1. and numerical_mask == 0.:
                        color = "red"
                        alpha_mask = 1.
                    if symbol_mask == 0. and numerical_mask == 1.:
                        color = "black"
                        alpha_mask = 1.
                    if symbol_mask == 0. and numerical_mask == 0.:
                        color = "white"
                        alpha_mask = 0.
                    if mask == True:
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * model.mask[l][i].item() * model.mask[l + 1][j].item())
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * model.mask[l][i].item() * model.mask[l + 1][j].item())
                    else:
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * y0, (l + 1 / 2) * y0 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [(l + 1 / 2) * y0 + y1, (l + 1) * y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)

    plt.xlim(0, 1)
    plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    plt.axis('off')

    # plot splines
    for l in range(neuron_depth - 1):
        n = width[l]
        for i in range(n):
            n_next = width[l + 1]
            N = n * n_next
            for j in range(n_next):
                id_ = i * n_next + j
                im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
                up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
                # newax = fig.add_axes([left, bottom, right - left, up - bottom])
                # print("ADD AXES COORDS:", [left, bottom, right - left, up - bottom])
                # new:
                increase = 0.5*(right - left)
                increase = 0.
                newax = fig.add_axes([left-increase, bottom-increase, right - left + 2*increase, up - bottom + 2*increase])
                # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                if mask == False:
                    newax.imshow(im, alpha=alpha[l][j][i])
                else:
                    ### make sure to run model.prune() first to compute mask ###
                    newax.imshow(im, alpha=alpha[l][j][i] * model.mask[l][i].item() * model.mask[l + 1][j].item())
                newax.axis('off')

    if in_vars is not None:
        n = model.width[0]
        print_norm_explicitly = True
        norms = [score.cpu().detach().numpy() for score in model.acts_scale]
        # [score2alpha(score.cpu().detach().numpy()) for score in model.acts_scale]
        norms = model.acts_scale[0].cpu().detach().numpy()# [0]
        norms = np.mean(model.acts_scale[0].cpu().detach().numpy(), axis=0)

        # print(norms, len(norms), n)
        for i in range(n):
            plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i], fontsize=tick_size, horizontalalignment='center', verticalalignment='center')
            if print_norm_explicitly:
                # this works so far only for the one layer KAN where one input corresponds to one spline
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.05, str(round(norms[i], 3)), fontsize=tick_size, horizontalalignment='center', verticalalignment='center')


    if out_vars is not None:
        n = model.width[-1]
        for i in range(n):
            plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), y0 * (len(model.width) - 1) + 0.1, out_vars[i], fontsize=1.5*tick_size, horizontalalignment='center', verticalalignment='center')


    # if title is not None:
    #     plt.gcf().get_axes()[0].text(0.5, y0 * (len(model.width) - 1) + 0.2, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')

