# Zeus 🕷️ — Autonomous AI Spider Robot

Zeus turns a [SunFounder PiCrawler](https://www.sunfounder.com/products/picrawler-kit) (a quadruped "spider" robot on a Raspberry Pi 5) into a **fully offline, voice-driven, face-tracking AI companion**. Say a wake word, ask it something, and it answers out loud and acts it out — walking, waving, doing push-ups, looking around — with the entire brain (speech-to-text → LLM → text-to-speech) running **locally on the Pi**. An optional OpenAI fallback kicks in only when the internet is available and you want it.

This repo is the open-source release of that work, meant for **anyone who owns a PiCrawler or is building a similar Pi-based robot from scratch.** Take the whole thing, or lift the parts you need (the brownout-safe movement tests, the on-device LLM benchmarks, the wake-word/STT loop).

> ⚠️ **Heads up:** the hardware libraries (`picrawler`, `robot_hat`, `vilib`, `sunfounder-controller`) are **SunFounder's**, under their own licenses, and are **not** included here — install them from SunFounder (see [Setup](#setup)). This repo contains only original Zeus code.

---

## What it does

- 🎙️ **Offline wake-word + speech recognition** — Vosk STT, always listening for `computer`, `spider`, `zeus`, or `picrawler`.
- 🧠 **On-device LLM** — Ollama running `qwen2.5:1.5b` (text) and `moondream:1.8b` (vision), tuned hard for Pi 5 latency. Optional `gpt-4o-mini` fallback when online.
- 🗣️ **Local text-to-speech** — Piper (`en_US-lessac-medium`), streamed sentence-by-sentence so it starts talking before the full answer is generated.
- 👁️ **Real-time face tracking** — `vilib` vision + custom yaw/pitch servo control with dead zones and adaptive smoothing.
- 🦿 **Embodied actions** — the LLM replies with structured actions (`wave_hand`, `push_up`, `fighting`, `nod`, `look_left`…) that execute as they stream in.
- 🛡️ **Brownout-safe by design** — movement is gated on live `vcgencmd` voltage/throttle checks; kill-switch flag files (`~/.zeus_no_move`, `~/.zeus_no_track`) disable motion for safe bench testing.
- 🔋 Runs headless as a `systemd` service on boot.

## Architecture

```
        ┌──────────┐   wake word   ┌──────────┐   text    ┌─────────────────┐
 mic ──▶│  Vosk STT │─────────────▶│ command  │──────────▶│  LLM router     │
        └──────────┘               │ capture  │           │ Ollama (local)  │
                                   └──────────┘           │ ↳ OpenAI (online)│
        ┌──────────┐                                       └────────┬────────┘
camera ▶│  vilib    │── face xy ─▶ yaw/pitch servo tracking          │
        └──────────┘                                        actions + answer
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                          ▼
                 ┌─────────────────┐                        ┌────────────────┐
                 │ PiCrawler gait/  │                        │  Piper TTS     │
                 │ preset_actions   │                        │  (speaker)     │
                 └─────────────────┘                        └────────────────┘
```

## Repo layout

| Path | What's inside |
|------|---------------|
| [`zeus.py`](zeus.py) | The main application — STT, wake word, LLM routing, TTS, face tracking, action execution. Runs as the `zeus` service. |
| [`examples/`](examples/) | Standalone demos: a movement showcase and a minimal voice→Ollama loop. Good first things to run. |
| [`benchmarks/`](benchmarks/) | On-Pi LLM latency/contract benchmarks — how the model choices above were picked. |
| [`tests/`](tests/) | Hardware & pipeline self-tests, including brownout-gated motion checks and no-human audio loopback tests. |

Each folder has its own README.

## Hardware

- **SunFounder PiCrawler** kit (Robot HAT + 12× servos, quadruped chassis)
- **Raspberry Pi 5** (this is tuned for the Pi 5; a Pi 4 works but is slower)
- Pi Camera, a USB/I²S mic, and a speaker
- A solid battery/power supply — under-powering the Pi while 12 servos move **will** brown it out (hence all the throttle-gating in `tests/`)

## Setup

```bash
# 1. SunFounder libraries (NOT in this repo — install per SunFounder's docs)
#    https://docs.sunfounder.com/projects/picrawler/en/latest/
#    Provides: picrawler, robot_hat, vilib, sunfounder-controller

# 2. Python deps for Zeus
pip install -r requirements.txt

# 3. Local AI runtimes
#    Ollama:  https://ollama.com
ollama pull qwen2.5:1.5b
ollama pull moondream:1.8b
#    Piper TTS voice: en_US-lessac-medium  (https://github.com/rhasspy/piper)
#    Vosk model: vosk-model-small-en-us-0.15  (https://alphacephei.com/vosk/models)
#      → unpack into /home/pi/ so zeus.py can find it

# 4. (optional) online fallback
export OPENAI_API_KEY=sk-...        # never hard-code it; see .env.example

# 5. Run
sudo python3 zeus.py                 # sudo needed for servo/GPIO access
```

See [`.env.example`](.env.example) for configurable environment variables.

### Run on boot (systemd)

Zeus is designed to run as a service. A minimal unit:

```ini
# /etc/systemd/system/zeus.service
[Unit]
Description=Zeus AI Spider Robot
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/zeus.py
WorkingDirectory=/home/pi
User=root
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now zeus
```

> **Bench-testing tip:** stop the service (`sudo systemctl stop zeus`) before running anything in `tests/`, so only one process owns the hardware. Drop a `~/.zeus_no_move` file to disable all motion while you work on the audio/LLM side.

## Safety notes

- The PiCrawler is strong and fast. Run motion tests with the robot on a stand or clear space.
- Zeus checks Pi voltage before/after moves and aborts on brownout — keep that behavior if you fork it.
- Servo work needs root; be deliberate about what runs as root.

## License

[MIT](LICENSE) © 2026 Gideon Glago. SunFounder libraries are excluded and remain under their own licenses.

## Acknowledgements

Built on [SunFounder PiCrawler](https://www.sunfounder.com/products/picrawler-kit), [Vosk](https://alphacephei.com/vosk/), [Ollama](https://ollama.com), [Piper](https://github.com/rhasspy/piper), and the open models `qwen2.5`, `moondream`, and `llama3.2`.
