
import torch
import torch.nn as nn
import numpy as np
from distutils.util import strtobool
from fake_spike_quantization import FakeSpikeQuantization


class TraditionalSpikingModel(nn.Module):
    def __init__(self, options, inp_dim):
        super(TraditionalSpikingModel, self).__init__()
        # Reading parameters
        self.input_dim = inp_dim
        self.hidden_size = int(options["hidden_size"])
        self.num_layers = int(options["num_layers"])
        self.use_bn = strtobool(options["use_bn"])
        self.threshold = float(options["threshold"])
        self.n_bits = int(options["n_bits"])
        self.dropout = float(options["dropout"])
        self.out_dim = int(options["output_size"])
        self.momentum = float(options["momentum"])
        self.to_do = options["to_do"]
        self.learn_tau = (options["learn_tau"] == "True")

        self.fsq_x = FakeSpikeQuantization(num_bits=self.n_bits, threshold=self.threshold, ema_momentum=self.momentum)
        self.linear_ws = nn.ModuleList([
            nn.Linear(self.input_dim if i == 0 else self.hidden_size,
                      self.hidden_size, bias=not(self.use_bn)) for i in range(self.num_layers)])
        self.taus = nn.ParameterList([
            nn.Parameter(torch.ones([self.hidden_size]), requires_grad=self.learn_tau) for i in range(self.num_layers)])
        if self.use_bn:
            self.bn_ws = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_size, momentum=self.momentum) for i in range(self.num_layers)])
            self.final_bn = nn.BatchNorm1d(self.out_dim, momentum=self.momentum)
        else:
            assert False, "Not support at the moment"
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
    
        outps = []
        vmem = torch.zeros_like(xs[0])
        for x in xs:
            # Compute outp
            vmem += x
            outp = self.fsq_x(vmem)
            outps.append(outp)
            # Update vmem
            vmem = vmem - outp
        outps = torch.stack(outps)
        inps = outps
        if self.to_do == 'forward':
            num_nonzeroact_inp = torch.sum((inps != 0), -1)
            num_nonzeroact_inps += num_nonzeroact_inp
            num_ops += (num_nonzeroact_inp * self.hidden_size)
            num_act_inps += torch.sum(torch.ones_like(inps, dtype=torch.long), -1)

        for i in range(self.num_layers):

            if self.training:
                dropout_mask = (torch.bernoulli(
                    torch.Tensor(batch_size, self.hidden_size).fill_(1 - self.dropout))/(1 - self.dropout)).to(target_device)

            ws = self.linear_ws[i](inps)

            if self.use_bn:
                ws_bn = self.bn_ws[i](ws.view(ws.shape[0] * ws.shape[1], ws.shape[2]))
                ws = ws_bn.view(ws.shape[0], ws.shape[1], ws.shape[2])
            
            prev_syn = torch.zeros((batch_size, self.hidden_size)).to(target_device)
            vmem = torch.zeros((batch_size, self.hidden_size)).to(target_device)
            outps = []
            for t in range(seq_len):
                if self.training:
                    syn = ws[t] * dropout_mask + self.taus[i] * prev_syn
                else:
                    syn = ws[t] + self.taus[i] * prev_syn

                # Compute outp
                vmem += syn
                outp = self.fsq_hs[i](vmem)

                outps.append(outp)

                prev_syn = syn
                vmem = vmem - outp
            outps = torch.stack(outps)
            inps = outps
            if self.to_do == 'forward':
                num_nonzeroact_rcr = torch.sum((inps != 0), -1)
                num_nonzeroact_rcrs += num_nonzeroact_rcr
                if i+1 == self.num_layers:
                    num_ops += (num_nonzeroact_rcr * self.out_dim) + 6 * self.hidden_size
                else: 
                    num_ops += (num_nonzeroact_rcr * self.hidden_size) + 6 * self.hidden_size
                num_act_rcrs += torch.sum(torch.ones_like(inps, dtype=torch.long), -1)

        ws = self.final_linear(inps)
        if self.use_bn:
            ws = self.final_bn(ws.view(seq_len * batch_size, -1))
        else:
            assert False, "Not support at the moment"

        ws = self.final_act(ws)
        
        if self.to_do == 'forward':
            return ws, num_nonzeroact_inps.view(seq_len * batch_size, -1), num_act_inps.view(seq_len * batch_size, -1), num_nonzeroact_rcrs.view(seq_len * batch_size, -1), num_act_rcrs.view(seq_len * batch_size, -1), num_ops.view(seq_len * batch_size, -1)/batch_size
        else:
            return ws

