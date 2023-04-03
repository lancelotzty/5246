import numpy as np
import random
import torch
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


mental_health_groups = [
    'EDAnonymous',
    'addiction',
    'alcoholism',
    'adhd',
    'anxiety',
    'autism',
    'bipolarreddit',
    'bpd',
    'depression',
    'healthanxiety',
    'lonely',
    'ptsd',
    'schizophrenia',
    'socialanxiety',
    'suicidewatch'
]

non_mental_health = [
    'conspiracy',
    'divorce',
    'fitness', 
    'guns', 
    'jokes', 
    'legaladvice', 
    'meditation', 
    'parenting', 
    'personalfinance', 
    'relationships', 
    'teaching',
]

len(mental_health_groups) + len(non_mental_health)

