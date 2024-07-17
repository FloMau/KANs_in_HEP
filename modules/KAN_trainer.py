import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from kan.LBFGS import *
from copy import deepcopy

def fit_with_early_stopping_batch(model, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
                            small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu', singularity_avoiding=False, y_th=1000., reg_metric='fa', patience=5, save_path='best_model.pt'):

    if lamb > 0. and not model.save_plot_data:
        print('setting lamb=0. If you want to set lamb > 0, set model.save_plot_data=True')
    
    def reg(acts_scale):
        reg_ = 0.
        for i in range(len(acts_scale)):
            vec = acts_scale[i]
            l1 = torch.sum(vec)
            p_row = vec / (torch.sum(vec, dim=1, keepdim=True) + 1)
            p_col = vec / (torch.sum(vec, dim=0, keepdim=True) + 1)
            entropy_row = - torch.mean(torch.sum(p_row * torch.log2(p_row + 1e-4), dim=1))
            entropy_col = - torch.mean(torch.sum(p_col * torch.log2(p_col + 1e-4), dim=0))
            reg_ += lamb_l1 * l1 + lamb_entropy * (entropy_row + entropy_col)

        for i in range(len(model.act_fun)):
            coeff_l1 = torch.sum(torch.mean(torch.abs(model.act_fun[i].coef), dim=1))
            coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(model.act_fun[i].coef)), dim=1))
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

        return reg_

    pbar = tqdm(range(steps), desc='description', ncols=100)

    if loss_fn is None:
        loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    else:
        loss_fn = loss_fn_eval = loss_fn

    grid_update_freq = int(stop_grid_update_step / grid_update_num)

    if opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "LBFGS":
        optimizer = LBFGS(model.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    results = {'train_loss': [], 'test_loss': [], 'reg': []}
    if metrics:
        for metric in metrics:
            results[metric.__name__] = []

    if batch == -1 or batch > dataset['train_input'].shape[0]:
        batch_size = dataset['train_input'].shape[0]
        batch_size_test = dataset['test_input'].shape[0]
    else:
        batch_size = batch
        batch_size_test = dataset['test_input'].shape[0]

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    early_stop = False
    model_to_save = deepcopy(model)

    global train_loss, reg_

    def closure():
        global train_loss, reg_
        optimizer.zero_grad()
        pred = model.forward(dataset['train_input'][train_id].to(device), singularity_avoiding=singularity_avoiding, y_th=y_th)
        train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
        if model.save_plot_data:
            if reg_metric == 'act':
                reg_ = reg(model.acts_scale_spline)
            if reg_metric == 'fa':
                model.attribute()
                reg_ = reg(model.edge_actscale)
        else:
            reg_ = torch.tensor(0.)
        objective = train_loss + lamb * reg_
        objective.backward()
        return objective

    if save_fig:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

    # for epoch in pbar:
    #     train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
    #     test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

    #     if epoch % grid_update_freq == 0 and epoch < stop_grid_update_step and update_grid and epoch >= start_grid_update_step:
    #         model.update_grid_from_samples(dataset['train_input'][train_id].to(device))

    #     if opt == "LBFGS":
    #         optimizer.step(closure)

    #     if opt == "Adam":
    #         pred = model.forward(dataset['train_input'][train_id].to(device), singularity_avoiding=singularity_avoiding, y_th=y_th)
    #         train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
    #         if model.save_plot_data:
    #             if reg_metric == 'act':
    #                 reg_ = reg(model.acts_scale_spline)
    #             if reg_metric == 'fa':
    #                 model.attribute()
    #                 reg_ = reg(model.edge_actscale)
    #         else:
    #             reg_ = torch.tensor(0.)
    #         loss = train_loss + lamb * reg_
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     test_loss = loss_fn_eval(model.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

    #     if epoch % log == 0:
    #         pbar.set_description(f"train loss: {torch.sqrt(train_loss).cpu().detach().numpy():.2e} | test loss: {torch.sqrt(test_loss).cpu().detach().numpy():.2e} | reg: {reg_.cpu().detach().numpy():.2e}")
    for epoch in pbar:
        permutation = torch.randperm(dataset['train_input'].size()[0])
        train_loss_epoch = []
        for i in range(0, dataset['train_input'].size()[0], batch_size):
            batch_indices = permutation[i:i+batch_size]
            train_id = batch_indices
            
            if epoch % grid_update_freq == 0 and epoch < stop_grid_update_step and update_grid and epoch >= start_grid_update_step:
                model.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)
                train_loss_epoch = train_loss

            if opt == "Adam":
                pred = model.forward(dataset['train_input'][train_id].to(device), singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                if model.save_plot_data:
                    if reg_metric == 'act':
                        reg_ = reg(model.acts_scale_spline)
                    if reg_metric == 'fa':
                        model.attribute()
                        reg_ = reg(model.edge_actscale)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                train_loss_epoch.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        test_loss = loss_fn_eval(model.forward(dataset['test_input'].to(device)), dataset['test_label'].to(device))
        train_loss_epoch = torch.tensor(train_loss_epoch)

        if epoch % log == 0:
            pbar.set_description(f"train loss: {torch.sqrt(torch.mean(train_loss_epoch)).cpu().detach().numpy():.2e} | test loss: {torch.sqrt(test_loss).cpu().detach().numpy():.2e} | reg: {reg_.cpu().detach().numpy():.2e}")

        if metrics:
            for metric in metrics:
                results[metric.__name__].append(metric().item())

        results['train_loss'].append(torch.sqrt(torch.mean(train_loss_epoch)).cpu().detach().numpy())
        results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results['reg'].append(reg_.cpu().detach().numpy())

        # Check early stopping condition
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_epoch = epoch
            model_to_save = deepcopy(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            early_stop = True
            print(f"Early stopping triggered. Saving best model from epoch {best_epoch}")
            break

        if save_fig and epoch % save_fig_freq == 0:
            model.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title=f"Step {epoch}", beta=beta)
            plt.savefig(os.path.join(img_folder, f'{epoch}.jpg'), bbox_inches='tight', dpi=200)
            plt.close()

    torch.save(model_to_save.state_dict(), save_path)
    if early_stop:
        model.load_state_dict(torch.load(save_path))

    model.log_history('train')
    return results

