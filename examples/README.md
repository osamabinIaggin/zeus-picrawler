# Examples

Two small scripts to run before you touch `zeus.py`. Stop the service first
(`sudo systemctl stop zeus`) so they can have the hardware to themselves.

- **[`demo.py`](demo.py)** — runs through every PiCrawler gait and preset action and announces each one over TTS. The quickest way to confirm the robot both moves and talks. `sudo python3 demo.py`
- **[`voice_ollama.py`](voice_ollama.py)** — a stripped-down voice → Ollama → speech loop without the rest of the Zeus stack. Read this one if you just want the conversational core, or want to reuse it somewhere else.

These are deliberately simpler than the main app, so they're a gentler place to start.
