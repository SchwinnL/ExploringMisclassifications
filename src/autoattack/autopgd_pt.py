#FROM https://github.com/fra31/auto-attack

import numpy as np
import time
import torch

import torch.nn as nn
import torch.nn.functional as F

    
class APGDAttack():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda', sigma=False, early_stop=False):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.early_stop = early_stop
        self.sigma = sigma
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def dlr_loss_targeted(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        y_target = (x).sort(1)[1][:, -2]

        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        if self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        elif self.loss == 'dlr_targeted':
            criterion_indiv = self.dlr_loss_targeted
        elif self.loss == "ce_scaled":
            criterion_indiv = ce_scaled
        elif self.loss == "jitter":
            criterion_indiv = jitter_loss(norm=2)
        elif self.loss == "cw":
            criterion_indiv = cw_loss
        else:
            raise ValueError('unknowkn loss')
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                if "jitter" in self.loss:
                    loss_indiv = criterion_indiv(x_in, x_adv, logits, y, self.sigma)
                else:
                    loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                if self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                if self.early_stop:
                    stop_idxs = logits.argmax(1) == y_in
                    x_adv[stop_idxs] = x_adv_1[stop_idxs] + 0.
                else:
                    x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    if "jitter" in self.loss:
                        loss_indiv = criterion_indiv(x_in, x_adv, logits, y, self.sigma)
                    else:
                        loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                grad += torch.autograd.grad(loss, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
              
        return x_best, acc, loss_best, x_best_adv
    
    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2', 'linfl2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        pred = self.model(x)
        acc = pred.max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)
            
            if not cheap:
                raise ValueError('not implemented yet')
            
            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        if self.sigma is not False:
                            self.sigma = torch.ones((y_to_fool.size(0), 1), device='cuda', requires_grad=True, dtype=torch.float) * self.sigma
                        else:
                            self.sigma = False
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), time.time() - startt))
            
            return acc, adv
        
        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            
                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            
            return loss_best, adv_best

def cw_loss(logits, y):
    label_mask = torch.eye(10, device='cuda')[y]
    correct_logit = torch.sum(label_mask * logits, 1)
    wrong_logit = torch.max((1 - label_mask) * logits - 1e4 * label_mask, 1)
    loss = -torch.nn.functional.relu(correct_logit - wrong_logit.values + 50)
    return loss

def jitter_loss(norm):
    def loss(X, X_adv, z, y, sigma, alpha=10):
        B = X.shape[0]
        Y = torch.eye(10)[y].cuda()
        ############################## logit scaling ###########################
        z_scaled = z / torch.norm(z.view(B, -1), p=float("inf"), dim=1, keepdim=True) * alpha
        z_scaled = torch.softmax(z_scaled, dim=1)
        z_noisy = z_scaled + torch.randn_like(z_scaled, ) * sigma
        ############################## l2 loss #################################
        l2 = torch.norm((z_noisy - Y).view(B, -1), p=2, dim=1)
        ############################## perturbation magnitude ##################
        non_adversarial_mask = z.argmax(1) != y
        magnitude = torch.norm((X - X_adv).view(B, -1), p=norm, dim=1)
        masked_magnitude = torch.ones_like(l2)
        masked_magnitude[non_adversarial_mask] = magnitude[non_adversarial_mask]
        ############################## final loss ##############################
        loss = l2 / masked_magnitude
        return loss

    return loss

def ce_scaled(logits, y, targets=None):
    if targets is not None:
        y = targets
    logits = logits / torch.max(torch.abs(logits), 1, keepdim=True).values * 10
    output = F.cross_entropy(logits, y, reduce=False, reduction='none')
    if targets is not None:
        output = -output
    return output
