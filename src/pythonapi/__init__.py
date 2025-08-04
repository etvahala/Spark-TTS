from logging import getLogger
from pathlib import Path

from torch import (
    cuda as torch_cuda,
    device as torch_device,
    randint as torch_randint)

from pythonapi.cannedvoice import CannedVoice, get_canned_voice
from pythonapi.generate import generate_input_directives, inference_into_wav
from pythonapi.interfaces import (
    Content,
    Narrator,
    OutputSpec,
    TokenizedContent,
    VoiceMetadata
)
from pythonapi.tokenizer import (
    access_tokenizer,
    tokenize_content
)

_LOG = getLogger(__name__)

def _get_torch_device()-> torch_device:
    """
    Get the appropriate torch device (CUDA or CPU).
    """
    if torch_cuda.is_available():
        return torch_device("cuda:0")
    else:
        return torch_device("cpu")

def _get_root_dir() -> Path:
    """
    Get the root directory of the Spark-TTS package.
    """
    return Path(__file__).resolve().parent.parent.parent

def _try_seeding(narrator: Narrator | None, device: torch_device) -> int:
    seed: int | None = None
    if narrator is not None:
        if narrator.seed is not None:
            _LOG.info('Using the explicitly provided seed: %(seed)s', {'seed': narrator.seed})
            seed = narrator.seed
        elif narrator.voice_spec and isinstance(narrator.voice_spec, Path):
            seed = get_canned_voice(narrator.voice_spec).seed
            _LOG.info('Using seed from existing voice spec: %(seed)s', {'seed': seed})

    if seed is None:
        seed = int(torch_randint(low=0, high=4294967295, size=(1, 1), device=device))
        _LOG.info('No seed provided, using randomly generated seed: %(seed)s', {'seed': seed})

    from torch import manual_seed as torch_manual_seed
    from numpy import random as np_random

    np_random.seed(seed)
    torch_manual_seed(seed)
    if torch_cuda.is_available():
        torch_cuda.manual_seed(seed)
        torch_cuda.manual_seed_all(seed)

    return seed

def generate_wav(content: Content, output_dir: Path, narrator: Narrator | None) -> TokenizedContent:
    """
    Generate a WAV file using the TTS model
    
    Returns: collected results in a TokenizedContent instance.
    """

    device = _get_torch_device()
    seed = _try_seeding(narrator, device)

    root_dir = _get_root_dir()
    model_dir = root_dir / "pretrained_models" / "Spark-TTS-0.5B"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist. Please ensure the model is downloaded and placed in the correct directory.")
    
    from cli.SparkTTS import SparkTTS
    model = SparkTTS(model_dir, device)

    input_directives = generate_input_directives(
        model=model,
        narrator=narrator)
    input_directives.seed = seed

    tokenizer = access_tokenizer(model_dir)    
    tokenized_content = tokenize_content(tokenizer, content, input_directives)    

    tokenized_content.output_file = inference_into_wav(
        tokenized_content,
        model,
        output_spec=OutputSpec(output_dir=output_dir))
    
    narrator_json_path = tokenized_content.output_file.with_suffix('.json')
    CannedVoice(narrator_json_path).write(
        VoiceMetadata(
            lines=content.lines,
            seed=seed,
            input_directives=input_directives.text,
        )
    )
