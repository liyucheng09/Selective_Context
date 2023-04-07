import pickle
from context_manager import *

with open('data/arxiv/text_2/ArxivContextManager_ArxivContextManager_sent.pkl', 'rb') as f:
    data = pickle.load(f)