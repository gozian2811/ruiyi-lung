import torch
from torch import nn

class FocalLoss(nn.Module):
	def __init__(self, loss_exp=0, balancing=None):
		super(FocalLoss, self).__init__()
		self.loss_exp = loss_exp
		if balancing is not None:
			self.balancing = nn.Parameter(torch.Tensor(balancing), requires_grad=False)

	def forward(self, input, target):
		cross_entropy = nn.functional.cross_entropy(input, target, reduction='none')
		if self.loss_exp != 0:
			#modulation = (target.float()-nn.functional.softmax(input, dim=1)[:,1]).abs().pow(self.loss_exp)
			modulation = (1-nn.functional.softmax(input, dim=1)[:,target].diag()).pow(self.loss_exp)
		else:
			modulation = 1
		#balancing = (target.float()-self.balancing).abs() * 2
		if hasattr(self, 'balancing'):
			balancing = self.balancing[target] * len(self.balancing)
		else:
			balancing = 1
		loss = torch.mean(cross_entropy * modulation * balancing)
		return loss

class HybridLoss(nn.Module):
	def __init__(self, basiccriterion, weightlist=[1], activation=None):
		super(HybridLoss, self).__init__()
		self.criterion = basiccriterion
		self.weightlist = weightlist
		if activation is None:
			self.activation = lambda x:x
		else:
			self.activation = activation
	
	def forward(self, inputs, targets, norm=None):
		totalloss = 0
		if type(inputs)==list or type(inputs)==tuple:
			num_hinge = len(inputs)
			for i in range(num_hinge):
				totalloss += self.criterion(self.activation(inputs[i]), targets[i]) * self.weightlist[i]
		else:
			num_hinge = 2
			totalloss = self.criterion(self.activation(inputs), targets)
		if norm is not None:
			if isinstance(norm, tuple):
				for ni in range(len(norm)):
					weight = self.weightlist[num_hinge+ni]
					if isinstance(norm[ni], tuple):
						for specnorm in norm[ni]:
							totalloss += specnorm * weight
					else:
						totalloss += norm[ni] * weight
			else:
				totalloss += norm * self.weightlist[num_hinge]
		return totalloss
		
class MultiTaskDifferenceLoss(nn.Module):
	def __init__(self, weightlist=[1], activation=None):
		super(MultiTaskDifferenceLoss, self).__init__()
		self.weightlist = weightlist
		if activation is None:
			self.activation = lambda x:x
		else:
			self.activation = activation
	
	def forward(self, inputs, targets):
		totalloss = 0
		if type(inputs)==list:
			for i in range(len(inputs)):
				if self.weightlist[i] is not None:
					totalloss += nn.functional.mse_loss(self.activation(inputs[i].view(-1)), targets[i]) * self.weightlist[i]
		else:
			totalloss = nn.functional.mse_loss(self.activation(inputs.view(-1)), targets) * self.weightlist[0]
		#loss = torch.mean(totalloss)
		return totalloss
