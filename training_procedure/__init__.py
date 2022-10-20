from .train import train
from .evaluate import evaluate
from .prepare import prepare_train , prepare_model , init
from functools import partial
import os
import torch as th
import pickle

class Trainer:
	def __init__(self , C , logger):
		self.C = C
		self.logger = logger

		self.prepare_train 	= partial(prepare_train	, self)
		self.prepare_model 	= partial(prepare_model	, self)
		self.init 			= partial(init 			, self)

		self.train 			= partial(train			, self)		
		self.evaluate 		= partial(evaluate		, self)

		self.flags = {}
		self.split_info = None

class ResultLogger:
	def __init__(self, C):
		self.C = C

		self.result = {}
		self.result['Training Res'] = []
		self.result['Val Res'] = []
		self.result['Test Res'] = []
		self.result['Training Loss'] = []
		self.log_name = self.make_log_name()

	def make_log_name(self):
		name = '{}_{}_{}_h_{}_prop_{}_alp_{}_lam_{}_mlp_bef_{}_aft_{}_e_{}_lr_{}_L2_{}_ReLU_dropout_{}_inp_dropout_{}'.format(
			self.C.date,
			self.C.data,
			self.C.seed,
			self.C.hidden_size,
			self.C.prop_step,
			self.C.alp,
			self.C.lam,
			self.C.mlp_bef,
			self.C.mlp_aft,
			self.C.num_epoch,
			self.C.lr,
			self.C.weight_decay,
			self.C.dropout,
			self.C.inp_dropout
		)

		if self.C.learn_emb > 0:
			name += '_learn_emb_{}'.format(self.C.learn_emb)
		
		if self.C.residual:
			name += '_residual'
		
		return name
	
	def save_result(self):

		if not os.path.isdir('./result_data'):
			os.mkdir('./result_data')

		with open('result_data/' + self.log_name + '.pkl', 'wb') as f:
			pickle.dump(self.result, f)
		
		return

	def save_model(self, model):

		if not os.path.isdir('./models'):
			os.mkdir('./models')
		th.save(model.state_dict(), './models/'+self.log_name + '.pt')

		return

	def save_best_model(self, model):

		if not os.path.isdir('./models'):
			os.mkdir('./models')
		th.save(model.state_dict(), './models/'+self.log_name + '_best.pt')

		return