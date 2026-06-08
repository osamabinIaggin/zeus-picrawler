#!/usr/bin/env python3
"""Phase 3 LLM benchmark: tests a model against the EXACT contract zeus.py needs.

For each prompt: sends the real OLLAMA_SYSTEM_PROMPT with production options,
measures latency, and checks (1) valid JSON, (2) actions are real, (3) persona.
Usage: python3 benchmark_llm.py <model>
"""
import json, re, sys, time, requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama3.2:1b"

# Copied verbatim from zeus.py:436
SYSTEM = """\
You are Zeus, a spider robot. Always reply with exactly this JSON format:
{"actions":["action1"],"answer":"your reply here"}
- "actions" is an array (use [] if none)
- "answer" is a SHORT non-empty string, always say something
Example: {"actions":["wave_hand"],"answer":"Hey there human!"}
Actions: sit,stand,wave_hand,shake_hand,fighting,excited,play_dead,nod,shake_head,\
look_left,look_right,look_up,look_down,warm_up,push_up,\
forward,backward,turn_left,turn_right,swagger,creep,spin,\
crouch_rise,sway,bob,play_music
Match action to intent: strut→swagger, sneak→creep, spin→spin, bounce→bob.
Keep answer short. Be playful. Never leave answer empty.\
"""

VALID_ACTIONS = set("sit,stand,wave_hand,shake_hand,fighting,excited,play_dead,nod,"
                    "shake_head,look_left,look_right,look_up,look_down,warm_up,push_up,"
                    "forward,backward,turn_left,turn_right,swagger,creep,spin,"
                    "crouch_rise,sway,bob,play_music".split(","))

# (prompt, expected-action-or-None, what we're testing)
TESTS = [
    ("what's your name?",        None,        "PERSONA (the Sparkles failure)"),
    ("wave at me",               "wave_hand", "action map"),
    ("spin around",              "spin",      "action map"),
    ("turn left",                "turn_left", "action map"),
    ("sneak forward quietly",    "creep",     "synonym->creep"),
    ("tell me a short joke",     None,        "open answer, no action"),
]


def parse(raw):
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
    try:
        return json.loads(raw), True
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()), False
        except Exception:
            pass
    return None, False


def main():
    print(f"\n===== {MODEL} =====")
    lat = []
    json_ok = 0
    action_ok = 0
    action_total = 0
    persona_hits = 0
    persona_total = 0
    for prompt, expect, label in TESTS:
        t0 = time.time()
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "system": SYSTEM, "prompt": prompt, "stream": False,
            "options": {"num_predict": 80, "num_ctx": 1024, "num_thread": 4,
                        "temperature": 0.7},
            "keep_alive": -1,
        }, timeout=180)
        dt = time.time() - t0
        lat.append(dt)
        raw = (r.json().get("response") or "").strip()
        parsed, strict = parse(raw)
        ok_json = parsed is not None
        json_ok += ok_json
        acts = parsed.get("actions", []) if isinstance(parsed, dict) else []
        ans = parsed.get("answer", "") if isinstance(parsed, dict) else raw
        bad_acts = [a for a in acts if a not in VALID_ACTIONS]
        if expect is not None:
            action_total += 1
            if expect in acts:
                action_ok += 1
        if "name" in prompt:
            persona_total += 1
            if "zeus" in (ans or "").lower():
                persona_hits += 1
        flag = "JSON✓" if strict else ("json~" if ok_json else "JSON✗")
        hit = "" if expect is None else ("  act✓" if expect in acts else f"  act✗(want {expect})")
        bad = f"  BAD_ACTS={bad_acts}" if bad_acts else ""
        print(f"[{dt:4.1f}s {flag}]{hit}{bad}  «{prompt}»")
        print(f"        actions={acts}  answer={ans!r}")
    n = len(TESTS)
    print(f"\n  SUMMARY {MODEL}:")
    print(f"    latency: avg {sum(lat)/n:.1f}s  min {min(lat):.1f}s  max {max(lat):.1f}s")
    print(f"    valid JSON: {json_ok}/{n}")
    print(f"    correct action: {action_ok}/{action_total}")
    print(f"    knows it's Zeus: {persona_hits}/{persona_total}")


if __name__ == "__main__":
    main()
