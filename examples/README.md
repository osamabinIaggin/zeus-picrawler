# Examples

Standalone, easy-to-run starting points. Stop the `zeus` service first
(`sudo systemctl stop zeus`) so these own the hardware.

| File | What it does |
|------|--------------|
| [`demo.py`](demo.py) | Runs through every PiCrawler gait and preset action with spoken TTS announcements. The fastest way to confirm the robot moves and talks. Run: `sudo python3 demo.py` |
| [`voice_ollama.py`](voice_ollama.py) | A minimal, self-contained **voice → Ollama → speech** loop, without the full Zeus stack. Great for understanding (or reusing) just the conversational core. |

> These are deliberately simpler than [`zeus.py`](../zeus.py) — read them first if the
> main app feels like a lot.
