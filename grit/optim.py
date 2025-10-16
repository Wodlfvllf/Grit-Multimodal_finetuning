from torch.optim import Optimizer


class GritOptimizer(Optimizer):
	"""Wrapper that applies GRIT preconditioning before the underlying step."""

	def __init__(self, optimizer: Optimizer, grit_manager: 'GRITManager'):
		self.optimizer = optimizer
		self.grit_manager = grit_manager

	@property
	def state(self):
		return self.optimizer.state

	@property
	def param_groups(self):
		return self.optimizer.param_groups

	@param_groups.setter
	def param_groups(self, value):
		self.optimizer.param_groups = value

	def step(self, closure=None):
		# Skip if preconditioning already applied in trainer's optimizer_step
		if self.grit_manager.factors_are_ready and not getattr(self.grit_manager, "_preconditioned", False):
			self.grit_manager.precondition_gradients()
		self.optimizer.step(closure)

	def zero_grad(self, set_to_none: bool = False):
		self.optimizer.zero_grad(set_to_none=set_to_none)

	def add_param_group(self, param_group: dict):
		self.optimizer.add_param_group(param_group)

	def state_dict(self):
		return self.optimizer.state_dict()

	def load_state_dict(self, state_dict: dict):
		self.optimizer.load_state_dict(state_dict)

	def __repr__(self):
		return f"GritOptimizer({self.optimizer.__repr__()})"


