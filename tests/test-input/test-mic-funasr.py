#!/usr/bin/env python3

import argparse
import queue
import sys
import time
import json
from datetime import datetime
import numpy as np

import sounddevice as sd
from funasr import AutoModel

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
    description="Real-time speech recognition with Fun-ASR",
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
    default="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    help="model name or path (default: FunAudioLLM/Fun-ASR-MLT-Nano-2512)",
)
parser.add_argument(
    "--device-id",
    type=str,
    default="cpu",
    help="device: cpu, cuda:0 (default: cpu)",
)
args = parser.parse_args(remaining)

try:
    if args.samplerate is None:
        args.samplerate = 16000

    print("#" * 80)
    print("Fun-ASR real-time recognition")
    print(f"Model: {args.model}")
    print(f"Device: {args.device_id}")
    print("Press Ctrl+C to stop")
    print("#" * 80)

    print("Loading model...")
    model = AutoModel(
        model=args.model,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=args.device_id,
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

        phrase_start_time = None
        last_result = ""
        audio_buffer = b""

        while True:
            data = q.get()
            audio_buffer += data

            if len(audio_buffer) < args.samplerate * 0.1:
                continue

            current_time = time.time()

            res = model.generate(
                input=[audio_buffer],
                cache={},
                language="ru",
                use_itn=True,
            )

            if res and len(res) > 0:
                text = res[0].get("text", "").strip()

                if text:
                    if phrase_start_time is None:
                        phrase_start_time = current_time

                    if text != last_result:
                        latency = current_time - phrase_start_time if phrase_start_time else 0
                        print()
                        log_word(text, latency)
                        last_result = text

            if len(audio_buffer) > args.samplerate * 5:
                audio_buffer = audio_buffer[-args.samplerate:]

except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(1, type(e).__name__ + ": " + str(e))
