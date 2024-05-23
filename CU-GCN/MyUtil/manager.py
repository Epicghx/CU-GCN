import os.path
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from . import Metric, classification_accuracy
from MyUtil.Regularization import Regularization
import ipdb

class Manager(object):
    """Handles training and pruning."""

    def __init__(self,args, model, train_loader, test_loader):

        self.args = args

        self.model = model
        self.train_loader = train_loader
        self.test_loader   = test_loader
        self.alpha = 0.1

        self.device = torch.device('cuda:0')
        self.criterion = nn.CrossEntropyLoss()
        self.regu = Regularization(model, args.weight_decay, 2).cuda()    # weight_decay = 4e-5
        # self.criterion = nn.KLDivLoss(reduce=True, reduction='mean')
        self.crt1 = nn.CrossEntropyLoss(reduction='sum')
        self.crt2 = nn.MSELoss(reduction='mean')
        self.nll_loss = nn.NLLLoss()
        return

    def kl_categorical(p_logit, q_logit):
        p = F.softmax(p_logit, dim=1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=1)
                             - F.log_softmax(q_logit, dim=1)), 1)
        return torch.mean(_kl)


    def train(self, optimizer, epoch_idx):
        # Set model to training mode
        wup = np.min([1.0, (epoch_idx + 1) / 20])
        self.model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        # scaler = torch.cuda.amp.GradScaler()


        for batch_idx, data in enumerate(self.train_loader, 0):
            inputs, labels, labelsprob = data.to(self.device), data.y.to(self.device), data.yprob.to(self.device)
            num = torch.tensor([len(data.y)]).to(self.device)

            if self.args.mixup:
                mix_alpha = np.random.beta(2.0, 2.0)
            else:
                mix_alpha = np.random.beta(2.0, 2.0)

            y = inputs.y.cpu().numpy().astype(np.uint8)
            index = np.random.permutation(len(y))
            fea_dim = inputs.x.size(1)
            x = inputs.x.reshape(len(y), -1, fea_dim)
            y = np.eye(np.max(y)+1)[y]

            inputs.x = (mix_alpha*x + (1-mix_alpha)*x[index, :, :]).reshape(-1, fea_dim)
            # inputs.y = torch.tensor(mix_alpha*y + (1-mix_alpha)*y[index]).float().to(self.device)
            new_label = inputs.y[index]

            optimizer.zero_grad()
            # Do forward-backward.
            # forward + backward + optimize
            if 'CU' in self.args.arch:
                output, kld_loss, drop_rates = self.model(inputs, training=True)
                l2_reg = None
                block_index = 0
                for layer in range(self.args.num_layer):
                    l2_lay_reg = None
                    if layer == 0:
                        for param in self.model.gcs[str(block_index)].parameters():
                            if l2_lay_reg is None:
                                l2_lay_reg = (param ** 2).sum()
                            else:
                                l2_lay_reg += (param ** 2).sum()
                        block_index += 1

                    else:
                        for iii in range(self.args.block):
                            for param in self.model.gcs[str(block_index)].parameters():
                                if l2_lay_reg is None:
                                    l2_lay_reg = (param ** 2).sum()
                                else:
                                    l2_lay_reg += (param ** 2).sum()
                            block_index += 1
                    # ipdb.set_trace()
                    l2_lay_reg = (1 - drop_rates[layer]) * l2_lay_reg

                    if l2_reg is None:
                        l2_reg = l2_lay_reg
                    else:
                        l2_reg += l2_lay_reg
                KL_loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1),
                                  reduction='sum')  # good one
                # nll_loss = F.nll_loss(F.log_softmax(outputs, dim=0), labels)
                tot_loss = KL_loss + wup * kld_loss
                loss = tot_loss + l2_reg + self.alpha * self.model.edge_weight.norm(1)+ self.regu(self.model)

                # loss = tot_loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)
                loss.backward()
            elif self.args.arch == 'RGNN':
                output = self.model(inputs)
                # KL_loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1),
                #                    reduction='sum')  # good one
                # CE_loss = self.crt1(F.log_softmax(output, dim=1), labels.long())
                if self.args.mixup:
                    loss = mix_alpha * self.nll_loss(F.log_softmax(output, dim=1), labels.long()) + (1-mix_alpha) * self.nll_loss(F.log_softmax(output, dim=1), new_label.long())
                else:
                    loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1), reduction='sum')  # good one
                loss = loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)
                # scaler.scale(loss).backward()
                loss.backward()
            elif self.args.arch == 'SGC':
                output = self.model(inputs)
                # KL_loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1),
                #                    reduction='sum')  # good one
                # CE_loss = self.crt1(F.log_softmax(output, dim=1), labels.long())
                if self.args.mixup:
                    loss = mix_alpha * self.nll_loss(F.log_softmax(output, dim=1), labels.long()) + (1-mix_alpha) * self.nll_loss(F.log_softmax(output, dim=1), new_label.long())
                else:
                    loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1), reduction='sum')  # good one
                loss = loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)
                # scaler.scale(loss).backward()
                loss.backward()
            elif self.args.arch == 'GCN':
                output = self.model(inputs)
                # KL_loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1),
                #                    reduction='sum')  # good one
                # CE_loss = self.crt1(F.log_softmax(output, dim=1), labels.long())
                if self.args.mixup:
                    loss = mix_alpha * self.nll_loss(F.log_softmax(output, dim=1), labels.long()) + (1-mix_alpha) * self.nll_loss(F.log_softmax(output, dim=1), new_label.long())
                else:
                    loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1), reduction='sum')  # good one
                loss = loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)
                # scaler.scale(loss).backward()
                loss.backward()
            elif self.args.arch == 'S2GC':
                output = self.model(inputs)
                # KL_loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1),
                #                    reduction='sum')  # good one
                CE_loss = self.crt1(F.log_softmax(output, dim=1), labels.long())
                if self.args.mixup:
                    loss = mix_alpha * self.nll_loss(F.log_softmax(output, dim=1), labels.long()) + (1-mix_alpha) * self.nll_loss(F.log_softmax(output, dim=1), new_label.long())
                else:
                    loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(labelsprob, dim=1),
                                reduction='sum')  # good one
                loss = loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)
                # scaler.scale(loss).backward()
                loss.backward()
            labels = labels.squeeze().long()  # + torch.full( (labels.size(0), labels.size(1)),1e-10 )

            # scaler.step(optimizer)
            optimizer.step()

            # scaler.update()
            train_loss.update(loss, num)
            train_accuracy.update(classification_accuracy(output, labels), num)

        return train_accuracy.avg.item()

    #{{{ Evaluate classification
    def eval(self, biases=None):
        """Performs evaluation."""

        self.model.eval()

        test_loss = Metric('test_loss')
        test_accuracy = Metric('test_accuracy')
        best_acc = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                inputs, labels = data.to(self.device), data.y.to(self.device)
                labels = labels.squeeze().long()
                num = torch.tensor([len(data.y)]).to(self.device)
                # Do forward-backward.
                # forward + backward + optimize
                # outs = [None] * 20
                # for j in range(20):
                #     output, _, _ = self.model(inputs, False)
                #     outs[j] = output.cpu().data.numpy()
                if 'CU' in self.args.arch:
                    output, _, _ = self.model(inputs)
                elif self.args.arch == 'RGNN':
                    output = self.model(inputs)
                elif self.args.arch == 'SGC':
                    output = self.model(inputs)
                elif self.args.arch == 'GCN':
                    output = self.model(inputs)
                elif self.args.arch == 'S2GC':
                    output = self.model(inputs)
                # outs = np.stack(outs)
                # ipdb.set_trace()
                # test_loss.update(self.crt1(output, labels), num)
                test_accuracy.update(classification_accuracy(output, labels), num)

        return test_accuracy.avg.item()


    def save_checkpoint(self, save_folder, epoch_idx):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx)
        checkpoint = {
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, filepath)
        return
