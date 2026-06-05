from .han2han_tools import *

# Import Rust functions
try:
    from .han2han_tools import transcribe_rs, transcribe_batch_rs, has_hanja, hanjain_rs, is_all_hanja
    # Use Rust version as default
    transcribe = transcribe_rs
    translate = transcribe_rs
    hanjain = hanjain_rs  # Use Rust version!
except ImportError:
    # Fallback to Python version
    from .hanja2hangul import Hanja2Hangul
    hanhan = Hanja2Hangul()
    transcribe = hanhan.convert
    translate = hanhan.convert
    hanjain = hanhan.hanjain

from .hanja2hangul import Hanja2Hangul
hanhan = Hanja2Hangul()
jamo_sentence = hanhan.jamo_sentence
jamo_to_word = hanhan.jamo_to_word
hanhansplit = hanhan.hanhansplit
unihan = Hanja2Hangul(1)
uniconvert = unihan.convert
cjk = unihan.cjk

from .stopwords import stopwords

# from .gukhantok import preprocess, tokenize

__doc__ = han2han_tools.__doc__
if hasattr(han2han_tools, "__all__"):
    __all__ = han2han_tools.__all__