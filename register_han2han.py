#!/usr/bin/env python3
# coding: utf-8
"""Register Han2Han models with HuggingFace AutoClasses."""

from transformers import AutoConfig, AutoTokenizer

from han2han_config import Han2HanConfig
from han2han_tokenizer import Han2HanTokenizer

AutoConfig.register("han2han", Han2HanConfig)

# register Han2Han tokenizer (as fast tokenizer since it inherits from AlbertTokenizerFast)
AutoTokenizer.register(Han2HanConfig, Han2HanTokenizer)

# only register models if PyTorch is available
try:
    import torch
    from transformers import AutoModel
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForMultipleChoice,
        AutoModelForSeq2SeqLM,
    )
    from modeling_han2han_pytorch import (
        Han2Han,
        Han2HanForSequenceClassification,
        Han2HanForTokenClassification,
        Han2HanForQuestionAnswering,
        Han2HanForMultipleChoice
    )

    # register models
    AutoModel.register(Han2HanConfig, Han2Han)
    AutoModelForSequenceClassification.register(Han2HanConfig, Han2HanForSequenceClassification)
    AutoModelForTokenClassification.register(Han2HanConfig, Han2HanForTokenClassification)
    AutoModelForQuestionAnswering.register(Han2HanConfig, Han2HanForQuestionAnswering)
    AutoModelForMultipleChoice.register(Han2HanConfig, Han2HanForMultipleChoice)
    AutoModelForSeq2SeqLM.register(Han2HanConfig, Han2Han)

    print("Han2Han models and tokenizer registered with HuggingFace AutoClasses!")
except ImportError:
    print("Han2Han tokenizer registered with HuggingFace AutoClasses (PyTorch not available, models not registered)")