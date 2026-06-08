# media

Screenshots and clips for the README live here.

## Adding GIFs

Short clips of the robot doing things (waving, push-ups, following a face) are
the best demo there is. Here's the no-fuss way to grab them off the Pi.

Record a few seconds of video on the Pi:

```bash
# Pi 5 / libcamera
libcamera-vid -t 5000 --width 640 --height 480 -o clip.h264
# or just screen-record your phone pointed at the robot, honestly
```

Turn it into a reasonably small GIF (needs ffmpeg):

```bash
ffmpeg -i clip.h264 -vf "fps=12,scale=480:-1:flags=lanczos" wave.gif
```

Drop the result in this folder and reference it from the README, e.g.:

```markdown
![Zeus waving hello](media/wave.gif)
```

Keep them under a few MB so the repo doesn't balloon — 480px wide at 10–12 fps
is plenty for a README.
