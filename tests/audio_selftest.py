#!/usr/bin/env python3
"""Self-contained mic/STT pipeline test — no human voice needed.
Measures ambient noise floor and runs an acoustic loopback (play a known
phrase through the speaker, record it on the mic, transcribe with whisper)."""
import audioop
import subprocess
import time
import wave

SR    = 16000
MIC   = "plughw:3,0"
PIPER = "/home/pi/en_US-lessac-medium.onnx"
WBIN  = "/home/pi/whisper.cpp/build/bin/whisper-cli"
WMODEL = "/home/pi/whisper.cpp/models/ggml-base.en.bin"


def rec(sec, path):
    subprocess.run(["arecord", "-D", MIC, "-f", "S16_LE", "-r", str(SR),
                    "-c", "1", "-q", "-d", str(sec), path],
                   check=False, stderr=subprocess.DEVNULL)


def stats(path):
    w = wave.open(path)
    d = w.readframes(w.getnframes())
    w.close()
    fb = int(SR * 0.03) * 2
    rmss = [audioop.rms(d[i:i+fb], 2) for i in range(0, len(d) - fb, fb)] or [0]
    return min(rmss), sum(rmss) // len(rmss), max(rmss)


def whisper(path):
    out = subprocess.run([WBIN, "-m", WMODEL, "-f", path, "-l", "en",
                          "-nt", "-np", "-t", "4"],
                         capture_output=True, text=True)
    return " ".join(l.strip() for l in out.stdout.splitlines() if l.strip())


print("=== ambient noise floor (2s, silence) ===")
rec(2, "/tmp/st_amb.wav")
lo, mean, hi = stats("/tmp/st_amb.wav")
print(f"ambient RMS  min={lo}  mean={mean}  max={hi}")

print("\n=== acoustic loopback (play phrase + record + transcribe) ===")
PHRASE = "testing one two three four five"
subprocess.run(["piper", "--model", PIPER, "--output_file", "/tmp/st_play.wav"],
               input=PHRASE.encode(), check=False, capture_output=True)
p = subprocess.Popen(["arecord", "-D", MIC, "-f", "S16_LE", "-r", str(SR),
                      "-c", "1", "-q", "-d", "4", "/tmp/st_loop.wav"],
                     stderr=subprocess.DEVNULL)
time.sleep(0.3)
subprocess.run(["aplay", "-D", "robothat", "/tmp/st_play.wav"],
               check=False, stderr=subprocess.DEVNULL)
p.wait()
lo, mean, hi = stats("/tmp/st_loop.wav")
print(f"loopback RMS min={lo}  mean={mean}  max={hi}")
print(f"spoken phrase : {PHRASE!r}")
print(f"whisper heard : {whisper('/tmp/st_loop.wav')!r}")
