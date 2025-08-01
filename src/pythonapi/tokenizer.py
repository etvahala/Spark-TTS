from logging import getLogger
from pathlib import Path
from typing import Iterator
from pythonapi.interfaces import Constants, Content, InputDirectives, TokenizedContent

from transformers import AutoTokenizer

_LOG = getLogger(__name__)

try:
    _LOG.debug("Attempting to load NLTK punkt tokenizer...")
    from nltk.data import (
        find as nltk_find)
    from nltk.tokenize import sent_tokenize as sentence_tokenize
    nltk_find('tokenizers/punkt')
    nltk_find('tokenizers/punkt_tab')
except LookupError:
    _LOG.info("NLTK punkt tokenizer not found. Downloading...")
    from nltk import download as nltk_download
    nltk_download('punkt')
    nltk_download('punkt_tab')

def access_tokenizer(model_dir: Path) -> AutoTokenizer:
    """
    Access the tokenizer from the pretrained models directory.
    """
    tokenizer_dir = model_dir / "LLM"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer

def _iterate_segments(tokenizer: AutoTokenizer, text: str, segmentation_threshold: int) -> Iterator[str]:
    sentences = sentence_tokenize(text)
    token_count = 0
    sentences_in_segment = []
    for sentence in sentences:
        token_count += len(tokenizer.encode(sentence, add_special_tokens=False))
        if token_count > segmentation_threshold:
            _LOG.debug("Token count exceeded model max length, yielding segment.")
            yield ' '.join(sentences_in_segment)
            sentences_in_segment = [sentence]
            token_count = 0
        else:
            sentences_in_segment.append(sentence)
    if sentences_in_segment:
        _LOG.debug("Yielding final segment.")
        yield ' '.join(sentences_in_segment)

def tokenize_content(tokenizer: AutoTokenizer, content: Content, input_directives: InputDirectives) -> TokenizedContent:
    """
    Tokenize the lines of text in the Content instance.

    Args:
        content (Content): The content to tokenize.
    
    Returns:
        Content: A new Content instance with tokenized lines.
    """
    text = '\n'.join(content.lines)
    
    input_token_count = len(tokenizer.encode(input_directives.text, add_special_tokens=False))
    text_token_count = len(tokenizer.encode(text, add_special_tokens=False))
    segment_size = content.max_tokens_per_segment or Constants.DEFAULT_TOKENS_PER_SEGMENT
    _LOG.info("Token count for text content: %(text)d - max segment size: %(segment)d", {'text': text_token_count, 'segment': segment_size - input_token_count})

    tokenize_content = TokenizedContent(
        input_directives=input_directives,
        segment_iterator=_iterate_segments(
            tokenizer,
            text,
            segment_size - input_token_count
        )
    )
    return tokenize_content
