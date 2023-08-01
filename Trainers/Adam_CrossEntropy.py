import torch
import torch.nn as nn
import torch.optim as optim

class AC_Trainer:
    def __init__(self, model, device, betas, weight_decay, epochs, train_load, lr=0.001, label_smoothing=0, grad_clip=None):
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.lossF = nn.CrossEntropyLoss(label_smoothing=label_smoothing) # label_smoothing 
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def step(self, input_sequence, output_sequence_reference) -> float:
        """x  = input_sequence.to(self.device)
        y  = output_sequence_reference.to(self.device)
        self.optimizer.zero_grad()
        output_sequence = self.model.forward(x)
        loss = self.lossF(output_sequence, y)
        loss.backward()

        if self.grad_clip:
            nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.scheduler.step()
        return loss"""
        pass
    
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        pass
    
