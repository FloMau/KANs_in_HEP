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
import sympy
import PIL
from copy import deepcopy
import mplhep as hep 
plt.style.use([hep.style.ROOT])


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


def plot(model, folder="./figures", beta=3, mask=False, metric='backward', scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None, varscale=1.0):
    
    global Symbol

    # forward to obtain activations
    if model.acts == None:
        if model.cache_data == None:
            raise Exception('model hasn\'t seen any data yet.')
        model.forward(model.cache_data)

    if metric == 'backward':
        model.attribute()

    if not os.path.exists(folder):
        os.makedirs(folder)
    # matplotlib.use('Agg')

    # separate spline plots here
    depth = len(model.width) - 1
    for l in range(depth):
        w_large = 2.0
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l+1]):
                rank = torch.argsort(model.acts[l][:, i])
                fig, ax = plt.subplots(figsize=(w_large, w_large))

                num = rank.shape[0]

                #print(model.width_in[l])
                #print(model.width_out[l+1])
                symbolic_mask = model.symbolic_fun[l].mask[j][i]
                numeric_mask = model.act_fun[l].mask[i][j]
                if symbolic_mask > 0. and numeric_mask > 0.:
                    color = 'purple'
                    alpha_mask = 1
                if symbolic_mask > 0. and numeric_mask == 0.:
                    color = "red"
                    alpha_mask = 1
                if symbolic_mask == 0. and numeric_mask > 0.:
                    color = "black"
                    alpha_mask = 1
                if symbolic_mask == 0. and numeric_mask == 0.:
                    color = "white"
                    alpha_mask = 0

                if tick == True:
                    ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                    ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                    x_min, x_max, y_min, y_max = model.get_range(l, i, j, verbose=False)
                    plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                    plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                else:
                    plt.xticks([])
                    plt.yticks([])
                if alpha_mask == 1:
                    plt.gca().patch.set_edgecolor('black')
                else:
                    plt.gca().patch.set_edgecolor('white')
                plt.gca().patch.set_linewidth(1.5)
                # plt.axis('off')

                # dont need these plots ATM, reduce time
                # continue
                plt.plot(model.acts[l][:, i][rank].cpu().detach().numpy(), model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, lw=5)
                if sample == True:
                    plt.scatter(model.acts[l][:, i][rank].cpu().detach().numpy(), model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color=color, s=400 * scale ** 2)
                plt.gca().spines[:].set_color(color)

                '''lock_id = model.act_fun[l].lock_id[j * model.width[l] + i].long().item()
                if lock_id > 0:
                    im = plt.imread(f'{folder}/lock.png')
                    newax = fig.add_axes([0.15, 0.7, 0.15, 0.15])
                    plt.text(500, 400, lock_id, fontsize=15)
                    newax.imshow(im)
                    newax.axis('off')'''

                plt.savefig(f'{folder}/sp_{l}_{i}_{j}.pdf', bbox_inches="tight", dpi=400)
                plt.close()

    def score2alpha(score):
        return np.tanh(beta * score)

    
    if metric == 'forward_n':
        scores = model.acts_scale
    elif metric == 'forward_u':
        scores = model.edge_actscale
    elif metric == 'backward':
        scores = model.edge_scores
    else:
        raise Exception(f'metric = \'{metric}\' not recognized')

    alpha = [score2alpha(score.cpu().detach().numpy()) for score in scores]

    ### I don't get the right L_1 norms with the built-in methods... calculate here:
    ### get L1 explicitly:

    width_in = np.array(model.width_in)
    width_out = np.array(model.width_out)
    L1_norms = []

    for l in range(len(model.width) - 1):
        n = width_in[l]
        n_next = width_out[l + 1]
        layer_norms = np.zeros((n, n_next))
        for i in range(n):
            for j in range(n_next):
                y = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
                L1_norm = np.mean(np.abs(y))
                layer_norms[i, j] = L1_norm
        L1_norms.append(layer_norms)

    L1_norms_normalized = [layer / np.max(layer) for layer in L1_norms]
    L1_norms = np.array(L1_norms, dtype=object)
    L1_norms_normalized = np.array(L1_norms_normalized, dtype=object)
    # Now you can use the normalized L1 norms
    alpha = L1_norms_normalized

    # draw skeleton
    width = np.array(model.width)
    width_in = np.array(model.width_in)
    width_out = np.array(model.width_out)
    A = 1
    y0 = 0.3  # height: from input to pre-mult
    z0 = 0.1  # height: from pre-mult to post-mult (input of next layer)

    neuron_depth = len(width)
    min_spacing = A / np.maximum(np.max(width_out), 5)

    max_neuron = np.max(width_out)
    max_num_weights = np.max(width_in[:-1] * width_out[1:])
    y1 = 0.4 / np.maximum(max_num_weights, 5) # size (height/width) of 1D function diagrams
    y2 = 0.15 / np.maximum(max_neuron, 5) # size (height/width) of operations (sum and mult)

    fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * (y0+z0)))
    # fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

    # -- Transformation functions
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # -- Take data coordinates and transform them to normalized figure coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
    
    # plot scatters and lines
    for l in range(neuron_depth):

        n = width_in[l]

        # scatters
        for i in range(n):
            plt.scatter(1 / (2 * n) + i / n, l * (y0+z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')

        # plot connections (input to pre-mult)
        for i in range(n):
            if l < neuron_depth - 1:
                n_next = width_out[l+1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    color="black"
                    if mask == True:
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=alpha[l][j][i] * model.mask[l][i].item() * model.mask[l + 1][j].item())
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=alpha[l][j][i] * model.mask[l][i].item() * model.mask[l + 1][j].item())
                    else:
                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N], [l * (y0+z0), l * (y0+z0) + y0/2 - y1], color=color, lw=2 * scale, alpha=alpha[l][i][j] * alpha_mask) # l j i were the indices before
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next], [l * (y0+z0) + y0/2 + y1, l * (y0+z0)+y0], color=color, lw=2 * scale, alpha=alpha[l][i][j] * alpha_mask) # l j i were the indices before

        # plot connections (pre-mult to post-mult, post-mult = next-layer input)
        if l < neuron_depth - 1:
            n_in = width_out[l+1]
            n_out = width_in[l+1]
            mult_id = 0
            for i in range(n_in):
                if i < width[l+1][0]:
                    j = i
                else:
                    if i == width[l+1][0]:
                        if isinstance(model.mult_arity,int):
                            ma = model.mult_arity
                        else:
                            ma = model.mult_arity[l+1][mult_id]
                        current_mult_arity = ma
                    if current_mult_arity == 0:
                        mult_id += 1
                        if isinstance(model.mult_arity,int):
                            ma = model.mult_arity
                        else:
                            ma = model.mult_arity[l+1][mult_id]
                        current_mult_arity = ma
                    j = width[l+1][0] + mult_id
                    current_mult_arity -= 1
                    #j = (i-width[l+1][0])//model.mult_arity + width[l+1][0]
                plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out], [l * (y0+z0) + y0, (l+1) * (y0+z0)], color='black', lw=2 * scale)

        plt.xlim(0, 1)
        plt.ylim(-0.1 * (y0+z0), (neuron_depth - 1 + 0.1) * (y0+z0))

    plt.axis('off')

    ###################
    # plot splines:
    ###################

    ### dirty fix trying to overcome the issue of plotting discrete input features by building a new model and feeding a continuous dataset into it. will only work for models with grid range (0,1)
    print("WARNING: plotting KAN evaluated on pseudodataset")
    model_for_plotting = KAN(
        width=[w[0] for w in model.width],
        grid=model.grid,
        k=model.k,
    )
    model_for_plotting.load_state_dict(model.state_dict())  # Load the cloned state dict
    n_features = model.width_in[0]  # Change this according to your model's input features
    n_rows = 10000
    dataset = np.linspace(0, 1, n_rows)
    dataset = np.tile(dataset, (n_features, 1)).T
    dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
    model_for_plotting.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        model_for_plotting.forward(dataset_tensor)

    for l in range(neuron_depth - 1):
        n = width_in[l]
        for i in range(n):
            n_next = width_out[l + 1]
            N = n * n_next
            for j in range(n_next):

                # rank = torch.argsort(model.acts[l][:, i])
                rank = torch.argsort(model_for_plotting.acts[l][:, i])

                id_ = i * n_next + j
                # im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                # im = PIL.Image.open(f'{folder}/sp_{l}_{i}_{j}.pdf', formats=["pdf"])
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                bottom = DC_to_NFC([0, l * (y0+z0) + y0/2 - y1])[1]
                up = DC_to_NFC([0, l * (y0+z0) + y0/2 + y1])[1]

                if l == 0:
                    width = right - left
                    height = up - bottom
                    newax = fig.add_axes([left, bottom, width, height])
                # dirty fix to increase size of plots in deeper layers which in our case are less wide
                else:
                    width = 3 * (right - left)
                    height = 3 * (up - bottom)
                    newax = fig.add_axes([left-1/3*width, bottom-2/3*height, width, height])

                # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                # if mask == False:
                #     newax.imshow(im, alpha=alpha[l][j][i])
                # else:
                #     ### make sure to run model.prune_node() first to compute mask ###
                #     newax.imshow(im, alpha=alpha[l][j][i] * model.mask[l][i].item() * model.mask[l + 1][j].item())
                # print("x", model.acts[l][:, i][rank].cpu().detach().numpy())
                # print("y", model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy())
                # exit()

                ### notes for spline_postacts:
                # returned from self.act_fun in Multkan, third return value (postacts_numerical). There, initialised as self.act_fun[l] = KANLayer(in_dim, out_dim + n_added_nodes, num=self.grid, k=self.k)
                # third return value indeed corresponds to the y-value of the learned activation function evaluated at the x-values...
                # newax.plot(model.acts[l][:, i][rank].cpu().detach().numpy(), model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color="tab:red", lw=1)
                newax.plot(model_for_plotting.acts[l][:, i][rank].cpu().detach().numpy(), model_for_plotting.spline_postacts[l][:, j, i][rank].cpu().detach().numpy(), color="tab:red", lw=1)
                newax.tick_params(axis="both", which="both", direction="in", pad=0, labelsize=2, length=1, width=0.1, grid_linewidth=0.1)
                newax.tick_params(axis="both", which="minor", direction="in", pad=0, labelsize=1, length=0, width=0, grid_linewidth=0.1)

                for spine in newax.spines.values():
                    spine.set_linewidth(0.1)  # Set a very thin width
                    spine.set_color('black')  # Ensure the color is black


    if in_vars != None:
        n = model.width_in[0]
        norms_for_plot = L1_norms[0]
        norms_for_plot = np.mean(norms_for_plot, axis=1)

        print_norm_explicitly = True

        for i in range(n):
            if isinstance(in_vars[i], sympy.Expr):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'${latex(in_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
            else:
                plt.gcf().get_axes()[0].text(
                    1 / (2 * (n)) + i / (n), 
                    -0.03, 
                    in_vars[i], 
                    fontsize=40 * scale * varscale, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    rotation=30,  # Rotate the text by 30 degrees
                    rotation_mode='anchor'  # Anchor the rotation to keep it at the same x location
                )
                if print_norm_explicitly:
                    # this works so far only for the one layer KAN where one input corresponds to one spline
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n)-0.01, 0.07, f"{norms_for_plot[i]:.2f}", fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center', rotation=90, rotation_mode="anchor")

    if out_vars != None:
        n = model.width_in[-1]
        for i in range(n):
            if isinstance(out_vars[i], sympy.Expr):
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(model.width) - 1) + 0.15, f'${latex(out_vars[i])}$', fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')
            else:
                plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0+z0) * (len(model.width) - 1) + 0.03, out_vars[i], fontsize=40 * scale * varscale, horizontalalignment='center', verticalalignment='center')

    plot_sigmoid = True
    if plot_sigmoid:
        y_l = 0.785
        x_min, x_max = -11, 11
        if l == 0:
            y_l = 0.705
            x_min, x_max = -5.5, 5.5
        x_sigmoid = np.linspace(x_min, x_max, 1000)
        newax = fig.add_axes([0.4985, y_l, width, height])
        newax.plot(x_sigmoid, 1/(1+np.exp(-x_sigmoid)), color="tab:blue", linewidth=1)
        newax.tick_params(axis="both", which="both", direction="in", pad=0, labelsize=2, length=1, width=0.1, grid_linewidth=0.1)
        newax.tick_params(axis="both", which="minor", direction="in", pad=0, labelsize=1, length=0, width=0, grid_linewidth=0.1)
        for spine in newax.spines.values():
            spine.set_linewidth(0.1)  # Set a very thin width
            spine.set_color('black')  # Ensure the color is black

    if title != None:
        plt.gcf().get_axes()[0].text(0.5, (y0+z0) * (len(model.width) - 1) + 0.3, title, fontsize=40 * scale, horizontalalignment='center', verticalalignment='center')



def get_L1_norms(model):

    width_in = np.array(model.width_in)
    width_out = np.array(model.width_out)

    L1_norms = []

    for l in range(len(model.width) - 1):
        n = width_in[l]
        n_next = width_out[l + 1]
        layer_norms = np.zeros((n, n_next))
        for i in range(n):
            for j in range(n_next):
                y = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
                L1_norm = np.mean(np.abs(y))
                layer_norms[i, j] = L1_norm
        L1_norms.append(layer_norms)

    L1_norms = np.array(L1_norms, dtype=object)

    return L1_norms
