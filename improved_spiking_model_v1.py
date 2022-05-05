
import torch
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
from fake_spike_quantization import FakeSpikeQuantization


class ImprovedSpikingModelV1(nn.Module):
    def __init__(self, options, inp_dim):
        super(ImprovedSpikingModelV1, self).__init__()
        # Reading parameters
        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.use_bn = strtobool(options["use_bn"])
        self.do_edat = strtobool(options["do_edat"])
        if self.do_edat:
            self.threshold = float(options["threshold"])
            self.n_bits = int(options["n_bits"])
        self.dropout = float(options["dropout"])
        self.out_dim = int(options["output_size"])
        self.momentum = float(options["momentum"])
        self.to_do = options["to_do"]

        if self.do_edat:
            self.fsq_x = FakeSpikeQuantization(num_bits=self.n_bits, threshold=self.threshold, ema_momentum=self.momentum)
        self.linear_ws = nn.ModuleList([
            nn.Linear(self.input_dim if i == 0 else self.hidden_size,
                      2 * self.hidden_size, bias=not(self.use_bn)) for i in range(self.num_layers)])
        if self.use_bn:
            self.bn_zs = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_size, momentum=self.momentum) for i in range(self.num_layers)])
            self.bn_cs = nn.ModuleList([nn.BatchNorm1d(self.hidden_size, momentum=self.momentum) for i in range(self.num_layers)])
            self.final_bn = nn.BatchNorm1d(self.out_dim, momentum=self.momentum)
        else:
            assert False, "Not support at the moment"
        if self.do_edat:
            self.fsq_hs = nn.ModuleList([
                FakeSpikeQuantization(num_bits=self.n_bits, threshold=self.threshold, ema_momentum=self.momentum) for i in range(self.num_layers)])

        self.final_linear = nn.Linear(self.hidden_size, self.out_dim, bias=False)
        self.final_act = nn.LogSoftmax(dim=1)

        # Weight initialization 
        for i in range(self.num_layers):
            nn.init.kaiming_uniform_(self.linear_ws[i].weight, nonlinearity='sigmoid')

        nn.init.uniform_(self.final_linear.weight,
                         -np.sqrt(0.01 / (self.hidden_size + self.out_dim)),
                         np.sqrt(0.01 / (self.hidden_size + self.out_dim)))

    def forward(self, xs):
        target_device = xs.device
        seq_len = xs.size(0)
        batch_size = xs.size(1)
        
        if self.to_do == 'forward':
            num_nonzeroact_inps = torch.zeros(seq_len, batch_size).long().to(target_device)
            num_act_inps = torch.zeros(seq_len, batch_size).long().to(target_device)
            num_nonzeroact_rcrs = torch.zeros(seq_len, batch_size).long().to(target_device)
            num_act_rcrs = torch.zeros(seq_len, batch_size).long().to(target_device)
            num_ops = torch.zeros(seq_len, batch_size).long().to(target_device)
    
        sparse_xs = []
        if self.do_edat:
            accerr_x = torch.zeros_like(xs[0])
            for x in xs:
                # Compute sparse_x
                accerr_x += x
                sparse_x = self.fsq_x(accerr_x)
                sparse_xs.append(sparse_x)
                # Update accerr_x
                accerr_x = accerr_x - sparse_x
            sparse_xs = torch.stack(sparse_xs)
            if self.to_do == 'forward':
                num_nonzeroact_inp = torch.sum((sparse_xs != 0), -1)
                num_nonzeroact_inps += num_nonzeroact_inp
                num_ops += (2 * num_nonzeroact_inp * self.hidden_size)
                num_act_inps += torch.sum(torch.ones_like(sparse_xs, dtype=torch.long), -1)
        else:
            if self.to_do == 'forward':
                num_nonzeroact_inps += torch.sum((xs != 0), -1)
                num_act_inps += torch.sum(torch.ones_like(xs, dtype=torch.long), -1)

        for i in range(self.num_layers):

            if self.training:
                dropout_mask = (torch.bernoulli(
                    torch.Tensor(batch_size, self.hidden_size).fill_(1 - self.dropout))/(1 - self.dropout)).to(target_device)

            if self.do_edat:
                ws = self.linear_ws[i](sparse_xs)
            else:
                ws = self.linear_ws[i](xs)
            wzs, wcs = ws.chunk(2, -1)

            if self.use_bn:
                wzs_bn = self.bn_zs[i](wzs.view(wzs.shape[0] * wzs.shape[1], wzs.shape[2]))
                wzs = wzs_bn.view(wzs.shape[0], wzs.shape[1], wzs.shape[2])

                wcs_bn = self.bn_cs[i](wcs.view(wcs.shape[0] * wcs.shape[1], wcs.shape[2]))
                wcs = wcs_bn.view(wcs.shape[0], wcs.shape[1], wcs.shape[2])
            
            prev_h = torch.zeros((batch_size, self.hidden_size)).to(target_device)
            if self.do_edat:
                accerr_h = torch.zeros((batch_size, self.hidden_size)).to(target_device)
                sparse_h = torch.zeros((batch_size, self.hidden_size)).to(target_device)
                sparse_hs = []
                for t in range(seq_len):
                    z = torch.sigmoid(wzs[t])
                    c = nn.functional.relu(wcs[t])
                    if self.training:
                        h = ((1 - z) * c * dropout_mask) + (z * prev_h)
                    else:
                        h = ((1 - z) * c) + (z * prev_h)

                    # Compute accerr_h
                    accerr_h += h
                    sparse_h = self.fsq_hs[i](accerr_h)
                    sparse_hs.append(sparse_h)

                    prev_h = h
                    accerr_h = accerr_h - sparse_h
                sparse_hs = torch.stack(sparse_hs)
                prev_sparse_xs = sparse_xs
                sparse_xs = sparse_hs
                if self.to_do == 'forward':
                    num_nonzeroact_rcr = torch.sum((sparse_xs != 0), -1)
                    num_nonzeroprevact_rcr = torch.sum((prev_sparse_xs != 0), -1)
                    num_nonzeroact_rcrs += num_nonzeroact_rcr
                    if i+1 == self.num_layers:
                        num_ops += (num_nonzeroact_rcr * self.out_dim) + 2 * (num_nonzeroprevact_rcr * self.hidden_size) + 8 * self.hidden_size
                    else: 
                        num_ops += (2 * num_nonzeroact_rcr * self.hidden_size) + 2 * (num_nonzeroprevact_rcr * self.hidden_size) + 8 * self.hidden_size
                    num_act_rcrs += torch.sum(torch.ones_like(sparse_xs, dtype=torch.long), -1)
            else:
                hs = []
                for t in range(seq_len):
                    z = torch.sigmoid(wzs[t])
                    c = nn.functional.relu(wcs[t])
                    if self.training:
                        h = ((1 - z) * c * dropout_mask) + (z * prev_h)
                    else:
                        h = ((1 - z) * c) + (z * prev_h)
                    hs.append(h) 
                    prev_h = h
                hs = torch.stack(hs)
                xs = hs
                if self.to_do == 'forward':
                    num_nonzeroact_rcrs += torch.sum((xs != 0), -1)
                    num_act_rcrs += torch.sum(torch.ones_like(xs, dtype=torch.long), -1)

        if self.do_edat:
            ws = self.final_linear(sparse_xs)
            if self.use_bn:
                ws = self.final_bn(ws.view(seq_len * batch_size, -1))
            else:
                assert False, "Not support at the moment"
        else:
            if self.use_bn:
                ws = self.final_bn(self.final_linear(xs).view(seq_len * batch_size, -1))
            else:
                assert False, "Not support at the moment"

        ws = self.final_act(ws)
        
        if self.to_do == 'forward':
            return ws, num_nonzeroact_inps.view(seq_len * batch_size, -1), num_act_inps.view(seq_len * batch_size, -1), num_nonzeroact_rcrs.view(seq_len * batch_size, -1), num_act_rcrs.view(seq_len * batch_size, -1), num_ops.view(seq_len * batch_size, -1)/batch_size
        else:
            return ws

