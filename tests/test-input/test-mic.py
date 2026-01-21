#!/usr/bin/env python3

# prerequisites: as described in https://alphacephei.com/vosk/install and also python module `sounddevice` (simply run command `pip install sounddevice`)
# Example usage using Dutch (nl) recognition model: `python test_microphone.py -m nl`
# For more help run: `python test_microphone.py -h`

import argparse
import queue
import sys
import time

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

q = queue.Queue()
speech_start_time = None


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
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
            device = sd.default.device["input"]  # type: ignore
        else:
            device = args.device
        device_info = sd.query_devices(device, "input")
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(16000)  # type: ignore

    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None

    with sd.RawInputStream(
        samplerate=args.samplerate,
        blocksize=int(args.samplerate * 0.10),
        device=args.device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        print("#" * 80)
        print("Press Ctrl+C to stop the recording")
        print("#" * 80)
        print("samplerate= ", args.samplerate)
        print("blocksize=", args.samplerate * 0.10)

        rec = KaldiRecognizer(model, args.samplerate)
        # rec.SetWords(True)
        # rec.SetEndpointerMode(EndpointerMode.SHORT)

        SetLogLevel(1)
        while True:
            data = q.get()
            partial = rec.PartialResult()
            if partial and speech_start_time is None:
                try:
                    partial_json = eval(partial) if partial.startswith("{") else {}
                    if partial_json.get("partial"):
                        speech_start_time = time.time()
                except:
                    pass
            if rec.AcceptWaveform(data):
                result = rec.Result()
                print(result)
                if speech_start_time is not None:
                    latency = time.time() - speech_start_time
                    print(f"[Latency: {latency:.3f}s]")
                    speech_start_time = None
            if dump_fn is not None:
                dump_fn.write(data)

except KeyboardInterrupt:
    print("\nDone")
    parser.exit(0)
except Exception as e:
    parser.exit(1, type(e).__name__ + ": " + str(e))
