# Zeus — an AI spider robot that actually listens unlike my ex...

Zeus is what happens when you give a [SunFounder PiCrawler](https://www.sunfounder.com/products/picrawler-kit) — a four-legged "spider" robot running on a Raspberry Pi 5 — a voice, ears, and enough of a brain to hold a short conversation. The whole thing runs on the Pi itself, no cloud needed. You say a wake word, ask it something, and it answers out loud while acting it out: walking around, waving, doing push-ups, tilting its head like it understood you (it usually did).

I built this on my own PiCrawler and figured anyone with the same kit — or building a Pi robot from scratch — would rather have the code than start from a blank file. Take all of it, or just lift the parts you need. I would love some ideas to improve or extend it's current functionality.

Quick heads up before you dig in: the hardware libraries (`picrawler`, `robot_hat`, `vilib`, `sunfounder-controller`) are SunFounder's, not mine, so they aren't in this repo. You grab those from SunFounder (link down in [Setup](#setup)). Everything here is my own code.

## See it

Demo clips are on the way — waving, push-ups, and the slightly-too-attentive face-tracking head swivel. They'll live in [`media/`](media/), which already has a two-command guide for pulling footage off the Pi and turning it into a GIF.

<!-- Once clips are in media/, uncomment and rename: -->
<!-- ![Zeus waving hello](media/wave.gif) -->
<!-- ![Push-ups, unprompted](media/pushup.gif) -->

## What it actually does

- **Listens offline.** Vosk handles the speech-to-text, so it's always half-listening for `computer`, `spider`, `zeus`, or `picrawler`. Nothing gets shipped off to some server to figure out what you said.
- **Thinks locally.** Ollama running `qwen2.5:1.5b` for text and `moondream:1.8b` for vision. I spent a slightly unreasonable amount of time shaving milliseconds off this — that's what the [`benchmarks/`](benchmarks/) folder is about. There's an optional OpenAI fallback for when it's online and you want a sharper answer.
- **Talks back** with Piper TTS, streamed sentence by sentence so it starts speaking before it's done thinking. Same trick people use when they start a sentence without knowing how it ends.
- **Follows your face** with the camera and turns its head to track you. Mildly unsettling, works great.
- **Moves on purpose.** The model replies with structured actions (`wave_hand`, `push_up`, `fighting`, `nod`, `look_left`…) that fire as they stream in, so the words and the movement actually line up instead of the robot miming something it said five seconds ago.
- **Tries not to brown itself out.** Twelve servos pulling at once can drop a Pi 5's voltage off a cliff. Zeus checks the voltage before and after every move and aborts if it looks sketchy, and there are two kill-switch files for bench testing without the robot thrashing around. More on that below.
- **Starts on boot** as a systemd service.

## How it fits together

```
        ┌──────────┐   wake word   ┌──────────┐   text    ┌─────────────────┐
 mic ──▶│  Vosk STT │─────────────▶│ command  │──────────▶│  LLM router      │
        └──────────┘               │ capture  │           │ Ollama (local)   │
                                   └──────────┘           │ └ OpenAI (online)│
        ┌──────────┐                                       └────────┬─────────┘
camera ▶│  vilib    │── face xy ─▶ yaw/pitch head tracking          │
        └──────────┘                                        actions + answer
                                                                    │
                          ┌─────────────────────────────────────────┤
                          ▼                                          ▼
                 ┌─────────────────┐                        ┌────────────────┐
                 │ PiCrawler gait / │                        │  Piper TTS     │
                 │ preset_actions   │                        │  (speaker)     │
                 └─────────────────┘                        └────────────────┘
```

## What's in here

- **[`zeus.py`](zeus.py)** — the whole thing. STT, wake word, LLM routing, TTS, face tracking, action execution. This is what runs as the service. It's big; start with the examples if it looks like a wall.
- **[`examples/`](examples/)** — two small, standalone scripts to run first: a movement-and-talking demo, and a bare-bones voice→Ollama loop.
- **[`benchmarks/`](benchmarks/)** — how I picked the models and prompts. The Pi is the bottleneck, so I measured instead of guessing.
- **[`tests/`](tests/)** — hardware and pipeline checks, including the brownout-gated motion tests and audio tests that don't need you to say anything.

Every folder has its own README.

## What you need

- A SunFounder **PiCrawler** kit (the Robot HAT, twelve servos, the chassis)
- A **Raspberry Pi 5** — this is tuned for the 5; a Pi 4 runs it, just slower
- The Pi camera, a mic, and a speaker
- A power supply that can actually keep up. Under-powering the Pi while the servos move is the single most annoying way to waste an afternoon, which is exactly why there's so much voltage-checking in `tests/`.

## Setup

```bash
# 1. SunFounder's libraries (not in this repo — install per their docs)
#    https://docs.sunfounder.com/projects/picrawler/en/latest/
#    gives you: picrawler, robot_hat, vilib, sunfounder-controller

# 2. Zeus's Python deps
pip install -r requirements.txt

# 3. The local AI bits
#    Ollama: https://ollama.com
ollama pull qwen2.5:1.5b
ollama pull moondream:1.8b
#    Piper TTS voice en_US-lessac-medium: https://github.com/rhasspy/piper
#    Vosk model vosk-model-small-en-us-0.15: https://alphacephei.com/vosk/models
#      unpack it into /home/pi/ so zeus.py finds it

# 4. (optional) the online fallback — set it in your env, don't paste it in code
export OPENAI_API_KEY=sk-...

# 5. Go
sudo python3 zeus.py     # sudo because the servos and GPIO need it
```

There's a [`.env.example`](.env.example) with the handful of things you can configure.

## Running it on boot

Zeus is meant to live as a systemd service. Minimal version:

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

When you want to poke at things in `tests/`, stop the service first (`sudo systemctl stop zeus`) so two processes aren't fighting over the same servos. Drop an empty `~/.zeus_no_move` file and Zeus keeps talking but stops moving — handy when you're working on the audio side and don't want it doing push-ups next to your keyboard.

## Not breaking things

The PiCrawler is stronger and faster than it looks. A few things I learned the boring way:

- Run motion tests with the robot on a stand or some clear space. It will move further than you expect.
- Keep the brownout checks if you fork this. A Pi that resets mid-move is a Pi that corrupts something eventually.
- Servo control needs root, so be deliberate about what you're running as root.

## License

[MIT](LICENSE) — do what you want, just keep the notice. SunFounder's libraries aren't included and stay under their own licenses.

## Thanks

Standing on the shoulders of [SunFounder](https://www.sunfounder.com/), [Vosk](https://alphacephei.com/vosk/), [Ollama](https://ollama.com), [Piper](https://github.com/rhasspy/piper), and the open `qwen2.5`, `moondream`, and `llama3.2` models. I mostly just wired them together and taught the result to do push-ups.
