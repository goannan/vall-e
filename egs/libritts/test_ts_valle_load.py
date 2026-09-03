import sys, os
from unittest.mock import MagicMock
for mod in ['k2', 'k2.version', 'kaldialign']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.path.insert(0, '/home/wu25/mrnas04home/projects/vall-e')
import torch
from icefall.utils import AttributeDict
from valle.models import get_model
from valle.data.tokenizer import AudioTokenizer

print("[1/3] Loading checkpoint...")
ckpt = torch.load('exp/valle/epoch-40.pt', map_location='cpu', weights_only=False)
print("[2/3] Initializing VALL-E model...")
args = AttributeDict(ckpt)
model = get_model(args)
model.load_state_dict(ckpt['model'], strict=True)
print("[3/3] Initializing TraceableSpeech Tokenizer...")
ts_tok = AudioTokenizer(
    watermark_backend='traceablespeech',
    ts_checkpoint='/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/g_00150000',
    ts_config='/home/wu25/mrnas04home/projects/vall-e/traceableSpeech/config.json',
)
print("TraceableSpeech + VALL-E successfully loaded!")
