#!/usr/bin/env python3
"""Validate the command-capture path (record_command_wav VAD + whisper) without
needing a wake word. Plays a known phrase into the mic shortly after recording
starts, then reports whether the VAD captured it and what whisper transcribed.
Run with zeus.service STOPPED so the mic is free."""
import audioop
import subprocess
import threading
import time
import wave

SR     = 16000
MIC    = "plughw:3,0"
PIPER  = "/home/pi/en_US-lessac-medium.onnx"
WBIN   = "/home/pi/whisper.cpp/build/bin/whisper-cli"
WMODEL = "/home/pi/whisper.cpp/models/ggml-base.en.bin"
CMD_WAV = "/tmp/cct_cmd.wav"


# ---- mirror of zeus.record_command_wav (continuous capture) ----
def record_command_wav(path=CMD_WAV, max_sec=9.0, start_timeout=4.0,
                       silence_sec=1.3, min_speech_sec=0.3, echo_guard=0.4):
    frame_dt = 0.03
    frame_bytes = int(SR * frame_dt) * 2
    proc = subprocess.Popen(["arecord", "-D", MIC, "-f", "S16_LE", "-r", str(SR),
                             "-c", "1", "-q", "-t", "raw"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _read():
        b = proc.stdout.read(frame_bytes)
        return b if len(b) == frame_bytes else None

    frames, started, speech_total, silence_run, elapsed = bytearray(), False, 0.0, 0.0, 0.0
    floor = threshold = 0.0
    try:
        for _ in range(int(echo_guard / frame_dt)):
            if _read() is None:
                break
        noise = []
        for _ in range(10):
            b = _read()
            if b is None:
                break
            noise.append(audioop.rms(b, 2)); frames += b
        floor = (sum(noise) / len(noise)) if noise else 120.0
        threshold = max(floor * 2.0, 450.0)
        while elapsed < max_sec:
            b = _read()
            if b is None:
                break
            frames += b; elapsed += frame_dt
            if audioop.rms(b, 2) > threshold:
                started = True; speech_total += frame_dt; silence_run = 0.0
            else:
                silence_run += frame_dt
                if not started:
                    if elapsed >= start_timeout:
                        break
                elif speech_total >= min_speech_sec and silence_run >= silence_sec:
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            proc.kill()
    dur = len(frames) / 2 / SR
    print(f"[rec] floor={floor:.0f} thr={threshold:.0f} started={started} "
          f"speech={speech_total:.2f}s dur={dur:.2f}s")
    if not started or speech_total < min_speech_sec:
        return None
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes(bytes(frames))
    return path


def whisper(path):
    out = subprocess.run([WBIN, "-m", WMODEL, "-f", path, "-l", "en",
                          "-nt", "-np", "-t", "4"], capture_output=True, text=True)
    return " ".join(l.strip() for l in out.stdout.splitlines() if l.strip())


PHRASE = "please turn left then walk forward and tell me a short joke"
subprocess.run(["piper", "--model", PIPER, "--output_file", "/tmp/cct_play.wav"],
               input=PHRASE.encode(), check=False, capture_output=True)


def _play_later():
    time.sleep(1.2)   # let recording + echo-guard + floor calc begin
    subprocess.run(["aplay", "-D", "robothat", "/tmp/cct_play.wav"],
                   check=False, stderr=subprocess.DEVNULL)


print(f"spoken phrase : {PHRASE!r}")
threading.Thread(target=_play_later, daemon=True).start()
wav = record_command_wav()
if wav:
    print(f"whisper heard : {whisper(wav)!r}")
else:
    print("whisper heard : <VAD captured nothing>")
