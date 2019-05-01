import torch


def optimizer(optimizer: str, params: list, lr: float):
	opt = None
	
	if optimizer == 'sgd':
		opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0002)
	else:
		opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0002)
	
	return opt