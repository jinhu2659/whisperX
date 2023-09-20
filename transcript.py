import argparse
from pathlib import Path
import warnings

import numpy as np
import torch

from whisperx.asr import load_model
from whisperx.utils import (LANGUAGES, TO_LANGUAGE_CODE, get_writer, optional_float,
                    optional_int, str2bool)


def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio_dir", type=str, help="audio dir to transcribe")
    parser.add_argument("--model", default="large-v2", help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--device_index", default=0, type=int, help="device index to use for FasterWhisper inference")
    parser.add_argument("--batch_size", default=8, type=int, help="the preferred batch size for inference")
    parser.add_argument("--compute_type", default="float16", type=str, choices=["float16", "float32", "int8"], help="compute type for computation")

    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    # vad params
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min_speech_duration_ms", type=int, default=250)
    parser.add_argument("--max_speech_duration_s", type=int, default=15)
    parser.add_argument("--min_silence_duration_ms", type=int, default=750)
    parser.add_argument("--window_size_samples", type=int, default=512)
    parser.add_argument("--speech_pad_ms", type=int, default=50)
    

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")

    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")

    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")


    parser.add_argument("--print_progress", type=str2bool, default = False, help = "if True, progress will be printed in transcribe() and align() methods.")
    # fmt: on

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    batch_size: int = args.pop("batch_size")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    device_index: int = args.pop("device_index")
    compute_type: str = args.pop("compute_type")

    task : str = args.pop("task")

    vad_options = dict(
        threshold = args.pop("threshold"),
        min_speech_duration_ms = args.pop("min_speech_duration_ms"),
        max_speech_duration_s = args.pop("max_speech_duration_s"),
        min_silence_duration_ms = args.pop("min_silence_duration_ms"),
        window_size_samples = args.pop("window_size_samples"),
        speech_pad_ms = args.pop("speech_pad_ms"),
    )

    print_progress: bool = args.pop("print_progress")


    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"
    align_language = args["language"] if args["language"] is not None else "en" # default to loading english if not specified

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    asr_options = {
        "beam_size": args.pop("beam_size"),
        "patience": args.pop("patience"),
        "length_penalty": args.pop("length_penalty"),
        "temperatures": temperature,
        "compression_ratio_threshold": args.pop("compression_ratio_threshold"),
        "log_prob_threshold": args.pop("logprob_threshold"),
        "no_speech_threshold": args.pop("no_speech_threshold"),
        "condition_on_previous_text": False,
        "initial_prompt": args.pop("initial_prompt"),
        "suppress_tokens": [int(x) for x in args.pop("suppress_tokens").split(",")],
        "suppress_numerals": args.pop("suppress_numerals"),
    }

    
    # Part 1: VAD & ASR Loop
    # model = load_model(model_name, device=device, download_root=model_dir)
    model = load_model(model_name, device=device, device_index=device_index, compute_type=compute_type, language=args['language'],
                       asr_options=asr_options, max_length=vad_options['max_speech_duration_s'], task=task)
    
    def transcript(audio_path: Path):
        # >> VAD & ASR
        print(f">>Performing transcription for {audio_path}")
        result = model.transcribe(str(audio_path), batch_size=batch_size, print_progress=print_progress, vad_options=vad_options)
        writer = get_writer(output_format, str(audio_path.parent))
        result["language"] = align_language
        writer(result, str(audio_path), {})
    
    for p in Path(args.pop("audio_dir")).iterdir():
        if p.is_dir():
            print(f'Load audios in {p}')
            for _f in p.iterdir():
                if _f.suffix in [".wav", ".mp3", ".flac", "m4a"]:
                    transcript(_f)
        elif p.suffix in [".wav", ".mp3", ".flac", ".m4a"]:
            transcript(p)


if __name__ == "__main__":
    cli()
