#!/usr/bin/env python3

import argparse
import queue
import sys
import time
import json
from datetime import datetime
import numpy as np

import sounddevice as sd
from faster_whisper import WhisperModel

q = queue.Queue()


def log_word(word, latency):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] WORD: {word} (latency: {latency:.3f}s)")
    sys.stdout.flush()


def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l",
    "--list-devices",
    action="store_true",
    help="show list of audio devices and exit",
)
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description="Real-time speech recognition with faster-whisper",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser],
)
parser.add_argument(
    "-d", "--device", type=int_or_str, help="input device (numeric ID or substring)"
)
parser.add_argument("-r", "--samplerate", type=int, help="sampling rate (default: 16000)")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="bzikst/faster-whisper-large-v3-russian",
    help="model name or path (default: bzikst/faster-whisper-large-v3-russian)",
)
parser.add_argument(
    "--compute-type",
    type=str,
    default="int8",
    help="compute type: int8, float16, float32 (default: int8)",
)
args = parser.parse_args(remaining)

try:
    if args.samplerate is None:
        args.samplerate = 16000

    print("#" * 80)
    print("faster-whisper real-time recognition (CPU mode)")
    print(f"Model: {args.model}")
    print(f"Compute type: {args.compute_type}")
    print("Press Ctrl+C to stop")
    print("#" * 80)

    print("Loading model...")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
    )
    print("Model loaded!")

    blocksize = int(args.samplerate * 0.02)

    with sd.RawInputStream(
        samplerate=args.samplerate,
        blocksize=blocksize,
        device=args.device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("Listening... Speak now!")
        print("-" * 60)

        audio_buffer = b""
        phrase_start_time = None
        last_partial = ""
        last_result_time = None
        silence_frames = 0
        silence_threshold = int(1.5 * args.samplerate / blocksize)

        while True:
            data = q.get()
            audio_buffer += data

            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = model.transcribe(
                audio_array,
                language="ru",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            current_time = time.time()

            detected_language = info.language
            segments_list = list(segments)
            has_speech = len(segments_list) > 0

            if has_speech and phrase_start_time is None:
                phrase_start_time = current_time

            for segment in segments_list:
                segment_text = segment.text.strip()
                if segment_text and segment_text != last_partial:
                    latency = current_time - phrase_start_time if phrase_start_time else 0
                    sys.stdout.write(f"\r[Partial] {segment_text}" + " " * 40)
                    sys.stdout.flush()
                    last_partial = segment_text
                    last_result_time = current_time
                    silence_frames = 0

            if last_result_time:
                elapsed = current_time - last_result_time
                if elapsed > 1.0:
                    silence_frames += 1

            if len(audio_buffer) > args.samplerate * 3:
                audio_buffer = audio_buffer[-args.samplerate:]

except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(1, type(e).__name__ + ": " + str(e))
