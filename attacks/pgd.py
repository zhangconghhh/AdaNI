'''
    Projected Gradient Descent (PGD) attack
'''

import torch
import torch.nn.functional as F

def pgd(model, data, target, epsilon = 8/255, k=7, a=0.01, random_start=True,
               d_min=0, d_max=1):
    perturbed_data = data.clone()
    perturbed_data.requires_grad = True

    data_max = data + epsilon
    data_min = data - epsilon
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_data.data = data + perturbed_data.uniform_(-1*epsilon, epsilon)
            perturbed_data.data.clamp_(d_min, d_max)

    for _ in range(k):
        output = model( perturbed_data, target, mixBN=False )[0]
        loss = F.cross_entropy(output, target)

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()
        data_grad = torch.autograd.grad(loss, perturbed_data, retain_graph=False, create_graph=False)[0]
        
        with torch.no_grad():
            perturbed_data.data += a * torch.sign(data_grad)
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max),
                                            data_min)
    perturbed_data.requires_grad = False

    return perturbed_data
