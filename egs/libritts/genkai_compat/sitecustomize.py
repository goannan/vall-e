"""Import stubs for optional training-time dependencies absent on GENKAI.

The standard VALL-E trainer only needs Icefall checkpoint/distributed helpers
and precomputed text/audio tokens.  Package ``__init__`` files nevertheless
import k2, phonemizer, pypinyin, and old TraceableSpeech modules.  Stub only
those unavailable optional modules so every torch.multiprocessing child sees
the same import environment.
"""

import importlib.machinery
import sys
from unittest.mock import MagicMock


OPTIONAL_MODULES = [
    "k2",
    "k2.version",
    "kaldialign",
    "pypinyin",
    "pypinyin.contrib",
    "pypinyin.contrib.tone_convert",
    "phonemizer",
    "phonemizer.backend",
    "phonemizer.backend.espeak",
    "phonemizer.backend.espeak.language_switch",
    "phonemizer.backend.espeak.words_mismatch",
    "phonemizer.punctuation",
    "phonemizer.separator",
    "traceableSpeech",
    "traceableSpeech.env",
    "traceableSpeech.meldataset",
    "traceableSpeech.models",
    "traceableSpeech.watermark",
]


class StubModule(MagicMock):
    __file__ = "/dev/null"
    __version__ = "0.0.0"
    __build_type__ = "Release"
    __git_sha1__ = "none"
    __git_date__ = "none"
    with_cuda = False


for module_name in OPTIONAL_MODULES:
    if module_name in sys.modules:
        continue
    try:
        __import__(module_name)
    except Exception:
        module = StubModule()
        module.__name__ = module_name
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, None)
        sys.modules[module_name] = module

# Link child modules to their parent modules (e.g. k2.version on k2)
for module_name in OPTIONAL_MODULES:
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        if parent_name in sys.modules and module_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, sys.modules[module_name])

