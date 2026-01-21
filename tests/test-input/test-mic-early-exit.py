#!/usr/bin/env python3

import argparse
import queue
import sys
import time
import json
from datetime import datetime

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

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


def callback(indata, frames, time, status):
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
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser],
)
parser.add_argument(
    "-f",
    "--filename",
    type=str,
    metavar="FILENAME",
    help="audio file to store recording to",
)
parser.add_argument(
    "-d", "--device", type=int_or_str, help="input device (numeric ID or substring)"
)
parser.add_argument("-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="language model; e.g. en-us, fr, nl; default is en-us",
)
args = parser.parse_args(remaining)

try:
    if args.samplerate is None:
        if args.device is None:
            device = sd.default.device["input"]
        else:
            device = args.device
        device_info = sd.query_devices(device, "input")
        args.samplerate = int(device_info["default_samplerate"])

    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None

    blocksize = int(args.samplerate * 0.02)

    with sd.RawInputStream(
        samplerate=args.samplerate,
        blocksize=blocksize,
        device=args.device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("#" * 80)
        print("Early exit on 2+ words - recognition finishes faster")
        print("Press Ctrl+C to stop")
        print("#" * 80)

        rec = KaldiRecognizer(model, args.samplerate)

        SetLogLevel(-1)
        last_partial = ""
        phrase_start_time = None
        recognized_words = []
        waiting_for_final = False

        while True:
            data = q.get()
            start_time = time.time()

            partial = rec.PartialResult()
            if partial:
                try:
                    partial_json = json.loads(partial)
                    text = partial_json.get("partial", "").strip()
                    if text and text != last_partial:
                        if phrase_start_time is None:
                            phrase_start_time = start_time

                        new_words = text.split()
                        old_words = last_partial.split() if last_partial else []
                        for i, word in enumerate(new_words):
                            if i >= len(old_words) or word != old_words[i]:
                                if word not in recognized_words:
                                    word_latency = time.time() - phrase_start_time
                                    print()
                                    log_word(word, word_latency)
                                    recognized_words.append(word)

                        sys.stdout.write(f"\r[Partial] {text}" + " " * 40)
                        sys.stdout.flush()
                        last_partial = text

                        if len(new_words) >= 2 and not waiting_for_final:
                            print(f"\n>> Multi-word detected, forcing early exit...")
                            waiting_for_final = True

                except:
                    pass

            if rec.AcceptWaveform(data) or waiting_for_final:
                result = rec.FinalResult()
                result_json = json.loads(result)
                text = result_json.get("text", "").strip()
                if text:
                    latency = time.time() - phrase_start_time if phrase_start_time else 0
                    mode = "early" if waiting_for_final else "normal"
                    sys.stdout.write(f"\r[Final ({mode})] {text} (total latency: {latency:.2f}s)\n")
                    sys.stdout.flush()
                    phrase_start_time = None
                    last_partial = ""
                    recognized_words = []
                    waiting_for_final = False

            if dump_fn is not None:
                dump_fn.write(data)

except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(1, type(e).__name__ + ": " + str(e))
