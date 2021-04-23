import pickle
import torch
path_logits = "logits"
path_logits_darts = path_logits + "logits_intel"
path_logits1 = path_logits+"/logits1"
path_logits2 = path_logits+"/logits2"
path_logits3 = path_logits+"/logits3"
path_dataset = "data/intel/"


logits1 = pickle.load( open( path_logits1 + "/logits0.p", "rb" ) )
logits2 = pickle.load( open( path_logits2 + "/logits0.p", "rb" ) )
logits3 = pickle.load( open( path_logits3 + "/logits0.p", "rb" ) )

for i in range(len(logits1)):
	print(torch.add(logits1,logits2)[1])
