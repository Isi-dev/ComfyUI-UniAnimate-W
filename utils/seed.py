<<<<<<< HEAD
import torch
import random
import numpy as np


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
=======
import torch
import random
import numpy as np


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
>>>>>>> 626e7afc02230297b6f553675ea1c32c29971314
     torch.backends.cudnn.deterministic = True