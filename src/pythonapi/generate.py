from datetime import datetime
from json import loads as json_loads
from logging import getLogger
from pathlib import Path
from re import findall as re_findall
from typing import Tuple

from numpy import concatenate as numpy_concatenate
from scipy.signal import resample
from soundfile import write as soundfile_write
from torch import (
    no_grad as torch_no_grad,
    Tensor as torch_tensor)

from cli.SparkTTS import SparkTTS
from pythonapi.cannedvoice import get_canned_voice
from pythonapi.interfaces import InputDirectives, Narrator, NarratorVoiceSpec, OutputSpec, PredefinedVoice, TokenizedContent
from sparktts.utils.token_parser import GENDER_MAP, LEVELS_MAP, TASK_TOKEN_MAP, TokenParser

_LOG = getLogger(__name__)

def _fill_in_contents(segment: str, input_directives: InputDirectives) -> str:
    """
    Fill in the content with input directives.

    <|start_content|><|end_content|> is replaced with <|start_content|>segment<|end_content|>


    Args:
        segment (str): The text segment to fill.
        input_directives (str): The input directives to include.

    Returns:
        str: The filled content.
    """
    if input_directives.predefined_voice:
        if not input_directives.predefined_voice.prompt_as_text.rstrip().endswith(('.', '!', '?')):
            input_directives.predefined_voice.prompt_as_text += '.'
        segment = f'{input_directives.predefined_voice.prompt_as_text}{segment}'

    filled = input_directives.text.replace("<|start_content|><|end_content|>", f"<|start_content|>{segment}<|end_content|>")
    return filled

def _generate_voice_from_existing_data(model: SparkTTS, narrator_path: Path) -> Tuple[str, str, torch_tensor]:
    """
    Generate voice from existing data.

    Args:
        narrator_path (Path): Path to the existing narrator JSON file.

    Returns:
        str: The text content to be narrated.
    """
    if not narrator_path.exists():
        raise FileNotFoundError(f"Provided narrator path does not exist: {narrator_path}")
    wav_path = narrator_path.with_suffix('.wav')
    if not wav_path.exists():
        raise FileNotFoundError(f"Expected WAV file does not exist: {wav_path}")
    
    voice_details = get_canned_voice(narrator_path)
    prompt_as_text = '\n'.join(voice_details.lines)

    global_token_ids, semantic_token_ids = model.audio_tokenizer.tokenize(wav_path)
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    # Prepare the input tokens for the model
    semantic_tokens = "".join(
        [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
    )
    inputs = [
        TokenParser.task("tts"),
        "<|start_content|>",
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
    ]

    inputs = "".join(inputs)
    return inputs, prompt_as_text, global_token_ids

@torch_no_grad()
def _inference(
    model: SparkTTS,
    text: str,
    global_token_ids: torch_tensor | None = None,
    temperature: float = 0.8,
    top_k: float = 50,
    top_p: float = 0.95) -> torch_tensor:
    """
    Returns:
        torch.Tensor: Generated waveform as a tensor.
    """
    model_inputs = model.tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.model.generate(
        **model_inputs,
        max_new_tokens=3000,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    # Trim the output tokens to remove the input tokens
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the generated tokens into text
    predicts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Extract semantic token IDs from the generated text
    pred_semantic_ids = (
        torch_tensor([int(token) for token in re_findall(r"bicodec_semantic_(\d+)", predicts)])
        .long()
        .unsqueeze(0)
    )

    if global_token_ids is None:
        # Not using a predefined voice, extract global token IDs
        global_token_ids = (
            torch_tensor([int(token) for token in re_findall(r"bicodec_global_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        )

    # Convert semantic tokens back to waveform
    wav = model.audio_tokenizer.detokenize(
        global_token_ids.to(model.device).squeeze(0),
        pred_semantic_ids.to(model.device),
    )

    return wav

def generate_input_directives(
    model: SparkTTS,
    narrator: Narrator | None) -> InputDirectives:
    """Generate input directives for the TTS model"""
    narrator = narrator if narrator else Narrator()

    if isinstance(narrator.voice_spec, Path):
        _LOG.info("Using existing narrator voice from path: %s", narrator.voice_spec)
        text, prompt_as_text,global_token_ids = _generate_voice_from_existing_data(model, narrator.voice_spec)
        return InputDirectives(
            predefined_voice=PredefinedVoice(
                global_token_ids=global_token_ids,
                prompt_as_text=prompt_as_text
            ),
            text=text,
            seed=narrator.seed,
            temperature=narrator.temperature,
            top_k=narrator.top_k,
            top_p=narrator.top_p
        )

    _LOG.info("Generating a new narrator voice with spec")
    assert isinstance(narrator.voice_spec, NarratorVoiceSpec)
    voice_spec = narrator.voice_spec
    attribute_tokens = "".join(
        [
            TokenParser.gender(voice_spec.gender),
            #TokenParser.mel_value(10),
            TokenParser.mel_level(voice_spec.pitch),
            #TokenParser.pitch_var_value(narrator.pitch),
            #TokenParser.pitch_var_level(narrator.pitch),
            #TokenParser.loudness_value(narrator.pitch),
            #TokenParser.loudness_level(narrator.pitch),
            #TokenParser.speed_value(narrator.speed),
            TokenParser.speed_level(voice_spec.speed)]
    )

    control_tts_inputs = [
        TokenParser.task("controllable_tts"),
        "<|start_content|>",
        "<|end_content|>",
        "<|start_style_label|>",
        attribute_tokens,
        "<|end_style_label|>",
    ]

    return InputDirectives(
        text="".join(control_tts_inputs),
        temperature=narrator.temperature,
        top_k=narrator.top_k,
        top_p=narrator.top_p
    )

def inference_into_wav(
        tokenized_content: TokenizedContent,
        model: SparkTTS,
        output_spec: OutputSpec) -> Path:
    """
    Generate a WAV file using the TTS model.

    Returns: Path to the generated WAV file.
    """
    start = datetime.now()
    _LOG.debug("Starting inference into WAV...")
    wavs = []
    input_directives = tokenized_content.input_directives
    for segment in tokenized_content.segment_iterator:        
        directed_text = _fill_in_contents(segment, input_directives)
        with torch_no_grad():
            
            wav = _inference(
                model,
                directed_text,
                global_token_ids=(
                    input_directives.predefined_voice.global_token_ids
                    if input_directives.predefined_voice else None
                ),
                temperature=input_directives.temperature,
                top_k=input_directives.top_k,
                top_p=input_directives.top_p,
            )
            _LOG.debug("Generated WAV segment")
            wavs.append(wav)
    final_wav = numpy_concatenate(wavs, axis=0)

    target_samplerate = 24000
    original_samplerate = 16000
    num_samples = int(len(final_wav) * target_samplerate / original_samplerate)
    final_wav = resample(final_wav, num_samples)

    filename = output_spec.output_filename or f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    path = output_spec.output_dir / filename

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile_write(path, final_wav, samplerate=target_samplerate)
    _LOG.info(f"Wav generated: {path} in {datetime.now() - start} seconds")
    return path
