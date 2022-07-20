from chargpt.model.model import GPT
from chargpt.model.trainer import Trainer
from chargpt.model.utils import CfgNode, setup_logging, set_seed

# dataset
from chargpt.data.dataset import CharDataset

# import package 
import torch
import sys

# create generate function
def get_config():
    C = CfgNode()

    # system
    C.system = CfgNode()
    C.system.seed = 3407
    C.system.work_dir = './weight/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C


# get default config and overrides from the command line, if any
config = get_config()
config.merge_from_args(sys.argv[1:])
print(config)
setup_logging(config)
set_seed(config.system.seed)

# construct the training dataset
text = open('.//chargpt//data//young.txt', 'r', encoding="utf-8").read() # don't worry we won't run out of file handles
train_dataset = CharDataset(config.data, text)

# construct the model
config.model.vocab_size = train_dataset.get_vocab_size()
config.model.block_size = train_dataset.get_block_size()
model = GPT(config.model)

# construct the trainer object
trainer = Trainer(config.trainer, model, train_dataset)

# D:\streamlit\textgen\chargpt\weight\gpt-nano_young.pt
path = ".//chargpt//weight//gpt-nano_young.pt"
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
model.eval()

def generate(prompt='', num_samples=10, steps=20, do_sample=True):
  if prompt == '': 
    # to create unconditional samples...
    # huggingface/transformers tokenizer special cases these strings
    prompt = '<|endoftext|>'
  else:
    x = torch.tensor([train_dataset.stoi[s] for s in prompt], dtype=torch.long)[None,...].to(trainer.device)
  
  x = x.expand(num_samples, -1)
  y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)

  text = []
  for i in range(num_samples):
      completion = ''.join([train_dataset.itos[int(j)] for j in y[i]])
      # print('-'*80)
      # print(completion)
      text.append(completion)
  return text

# generate(prompt = "ร่างกายของเรา ")
