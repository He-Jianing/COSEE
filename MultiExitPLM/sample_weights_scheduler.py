#encoding:utf-8
import torch
from torch import nn




class SampleWeightsScheduler:
    def __init__(self,beta,total_step,num_layers):
        self.beta=beta
        self.total_step=total_step
        self.step_counter=0
        self.num_layers=num_layers
        self.weights_basis=torch.zeros(num_layers,num_layers)
        for i in range(len(self.weights_basis)):
            for j in range(len(self.weights_basis)):
                self.weights_basis[i][j]=abs(i-j)

    def get_sample_weights(self):
        current_beta=self.step_counter*self.beta/self.total_step
        sample_weights=-current_beta*self.weights_basis
        sample_weights = torch.softmax(sample_weights, dim=-1)

        return sample_weights

    def step(self):
        self.step_counter+=1