import torch.optim as optim
import spacy
import en_core_web_sm
import de_core_news_sm
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_eng = en_core_web_sm.load()
spacy_de = de_core_news_sm.load()



