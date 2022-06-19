import torch
import torch.nn.functional as F


class SoftCrossEntropy:
    def __call__(self, inputs, target):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1).mean()
        return loss
