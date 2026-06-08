#!/usr/bin/env python3
"""
Zeus Movement Demo — runs all actions with TTS announcement.
Run with: sudo python3 ~/demo.py
"""
import subprocess, sys, time, math
sys.path.append('/home/pi/picrawler/examples')
from picrawler import Picrawler
from preset_actions import (
    wave_hand, shake_hand, fighting, excited,
    play_dead, nod, shake_head, look_left, look_right,
    look_up, look_down, warm_up, push_up,
)

PIPER  = '/home/pi/en_US-lessac-medium.onnx'
STAND_Z = [-60, -60, -60, -60]

crawler = Picrawler()
time.sleep(1)
crawler.do_action('stand', speed=60)
time.sleep(1)
subprocess.run(['pinctrl', 'set', '20', 'op', 'dh'], check=False)

def say(text):
    print(f">> {text}")
    r = subprocess.run(['piper', '--model', PIPER, '--output_file', '/tmp/demo.wav'],
                       input=text.encode(), capture_output=True)
    if r.returncode == 0:
        subprocess.run(['aplay', '-D', 'robothat', '/tmp/demo.wav'], check=False)

def pause(n=1.5):
    time.sleep(n)

def lerp(a, b, t): return a + (b - a) * t

def apply_pose(yaw, z):
    crawler.do_step([
        [45 - yaw*0.3, 45 + yaw, z[0]],
        [45 + yaw*0.3, 45 - yaw, z[1]],
        [45 - yaw*0.3, 45 + yaw, z[2]],
        [45 + yaw*0.3, 45 - yaw, z[3]],
    ], speed=100)

def smooth_z(from_z, to_z, steps=20, delay=0.05):
    for i in range(steps + 1):
        t = i / steps
        apply_pose(0, [lerp(from_z[j], to_z[j], t) for j in range(4)])
        time.sleep(delay)

# ── EXISTING PRESET ACTIONS ────────────────────────────────────────────────────
say("Wave hand")
wave_hand(crawler); pause()

say("Shake hand")
shake_hand(crawler); pause()

say("Fighting stance")
fighting(crawler); pause()

say("Excited")
excited(crawler); pause()

say("Play dead")
play_dead(crawler); pause()
crawler.do_action('stand', speed=60); time.sleep(0.8)

say("Nod")
nod(crawler); pause()

say("Shake head")
shake_head(crawler); pause()

say("Look left")
look_left(crawler); pause()

say("Look right")
look_right(crawler); pause()

say("Look up")
look_up(crawler); pause()

say("Look down")
look_down(crawler); pause()

say("Warm up")
warm_up(crawler); pause()

say("Push up")
push_up(crawler); pause()

# ── WALKING ────────────────────────────────────────────────────────────────────
say("Walk forward, 4 steps")
for _ in range(4):
    crawler.do_action('forward', speed=80)
pause()

say("Walk backward, 4 steps")
for _ in range(4):
    crawler.do_action('backward', speed=80)
pause()

say("Turn left")
for _ in range(3):
    crawler.do_action('turn left', speed=80)
pause()

say("Turn right")
for _ in range(3):
    crawler.do_action('turn right', speed=80)
pause()

# ── NEW MOVEMENTS ──────────────────────────────────────────────────────────────

say("Crouch and rise")
smooth_z(STAND_Z, [-82,-82,-82,-82], steps=20, delay=0.05)
time.sleep(0.5)
smooth_z([-82,-82,-82,-82], STAND_Z, steps=30, delay=0.06)
pause()

say("Body sway left and right")
for _ in range(3):
    apply_pose(-18, STAND_Z); time.sleep(0.4)
    apply_pose( 18, STAND_Z); time.sleep(0.4)
apply_pose(0, STAND_Z); pause()

say("Swagger walk")
for _ in range(4):
    crawler.do_action('forward', speed=60)
pause()

say("Spin in place")
for _ in range(6):
    crawler.do_action('turn right', speed=100)
pause()

say("Bob up and down")
for _ in range(4):
    smooth_z(STAND_Z, [-48,-48,-48,-48], steps=10, delay=0.04)
    smooth_z([-48,-48,-48,-48], STAND_Z, steps=10, delay=0.04)
pause()

say("Creep forward slowly")
for _ in range(4):
    crawler.do_action('forward', speed=30)
pause()

say("That is everything I can do! Pretty impressive right?")
excited(crawler)
crawler.do_action('stand', speed=60)
