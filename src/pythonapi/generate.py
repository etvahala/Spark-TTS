from datetime import datetime
from logging import getLogger
from pathlib import Path
from re import findall as re_findall

from numpy import concatenate as numpy_concatenate
from soundfile import write as soundfile_write
from torch import (
    no_grad as torch_no_grad,
    Tensor as torch_tensor)

from cli.SparkTTS import SparkTTS
from pythonapi.interfaces import Constants, Narrator, OutputSpec, TokenizedContent
from sparktts.utils.token_parser import GENDER_MAP, LEVELS_MAP, TASK_TOKEN_MAP

_LOG = getLogger(__name__)

def _fill_in_contents(segment: str, input_directives: str) -> str:
    """
    Fill in the content with input directives.

    <|start_content|><|end_content|> is replaced with <|start_content|>segment<|end_content|>


    Args:
        segment (str): The text segment to fill.
        input_directives (str): The input directives to include.

    Returns:
        str: The filled content.
    """
    filled = input_directives.replace("<|start_content|><|end_content|>", f"<|start_content|>{segment}<|end_content|>")
    return filled

@torch_no_grad()
def _inference(
    model: SparkTTS,
    text: str,
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
        output_ids[len(input_ids) :]
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
    narrator: Narrator | None) -> str:
    """Generate input directives for the TTS model

        Return:
            str: Input prompt prefix
    """
    narrator = narrator if narrator else Narrator()

    gender_id = GENDER_MAP[narrator.gender]
    pitch_level_id = LEVELS_MAP[narrator.pitch]
    speed_level_id = LEVELS_MAP[narrator.speed]

    pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
    speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
    gender_tokens = f"<|gender_{gender_id}|>"

    attribute_tokens = "".join(
        [gender_tokens, pitch_label_tokens, speed_label_tokens]
    )

    control_tts_inputs = [
        TASK_TOKEN_MAP["controllable_tts"],
        "<|start_content|>",
        "<|end_content|>",
        "<|start_style_label|>",
        attribute_tokens,
        "<|end_style_label|>",
    ]

    return "".join(control_tts_inputs)

def inference_into_wav(tokenized_content: TokenizedContent, model: SparkTTS, output_spec: OutputSpec) -> Path:
    """
    Generate a WAV file using the TTS model.

    Returns: Path to the generated WAV file.
    """
    
    wavs = []
    for segment in tokenized_content.segment_iterator:
        directed_text = _fill_in_contents(segment, tokenized_content.input_directives)
        with torch_no_grad():
            
            wav = _inference(
                model,
                directed_text
            )
            wavs.append(wav)
    final_wav = numpy_concatenate(wavs, axis=0)


    path = output_spec.output_dir / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile_write(path, final_wav, samplerate=16000)
    _LOG.info(f"Wav generated: {path}")
