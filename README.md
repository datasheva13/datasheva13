# Moto Shorts

Turns raw GoPro off-road POV footage into cinematic, raw-sound shorts, every day.

**What it does each morning**
1. Downloads new rides from GoPro Cloud (or picks up files you drop in the `inbox` folder).
2. Finds the best moments: exhaust sound (revs, throttle hits, climbs) + scenery (open views, light, variety) + GPS (speed, climbs). It never repeats a moment or similar scenery on consecutive days.
3. Makes **3 shorts**: 9:16 vertical, sharp footage centered on a moving blurred copy of itself, slow subtle push-in, natural cinematic grade for HERO11 Flat footage, raw exhaust audio.
4. Writes moody hooks, title, description, **3 hashtags** and a **one-word thumbnail**.
5. When enough good footage has piled up, builds a **long 16:9 YouTube video** with chapters.
6. You review everything in a dashboard, tweak anything, click **Approve**, then post (about 30 seconds per platform).

**Exports:** YouTube 4K (2160×3840 shorts / 3840×2160 long), 10-bit HEVC ~100 Mbps · Instagram/TikTok 1080×1920 high bitrate · no watermarks.

---

## Setup (Windows, about 20 minutes, once)

### 1. Get the code
On GitHub open this repository → switch to branch `claude/motorcycle-pov-video-pipeline-kyhfk8` → **Code → Download ZIP**. Unzip it somewhere with space, e.g. `D:\MotoShorts`.

### 2. Run the installer
Open the folder, click the address bar, type `powershell`, press Enter. Then run:
```
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```
It installs Python, FFmpeg (with NVIDIA GPU encoding) and everything else. If it says *"close this window and run again"*, do exactly that (happens up to twice).

### 3. Add your keys (`.env` file in the folder, open with Notepad)
- **ANTHROPIC_API_KEY** (captions): sign up at console.anthropic.com → *API Keys* → *Create key* → paste it after `ANTHROPIC_API_KEY=`. Cost: a few cents a day.
- **GOPRO_ACCESS_TOKEN** (GoPro Cloud):
  1. In Chrome, go to **gopro.com/media-library** and sign in.
  2. Press **F12** → tab **Application** → left side **Cookies → https://gopro.com**.
  3. Find **gp_access_token**, double-click its *Value*, copy, and paste it after `GOPRO_ACCESS_TOKEN=`.
  
  This token expires every so often. When it does, the dashboard shows a red banner with a box to paste a new one. GoPro has no official API, so this is the unofficial route. If it ever breaks, drop files into the `inbox` folder instead (export from Quik or copy from the SD card).

Run the check again any time:
```
.venv\Scripts\python -m moto_shorts check
```

### 4. Phone folder for TikTok / Instagram
Install **Google Drive for desktop** (or iCloud/OneDrive), then in `config.yaml` set `handoff_dir:` to a folder inside it, e.g. `G:\My Drive\MotoShorts`. Every approved short lands there as `video.mp4` + `cover.jpg` + `caption.txt`, ready on your phone.

### 5. Turn on the daily run
```
powershell -ExecutionPolicy Bypass -File scripts\install_windows_task.ps1 -Time 06:15
```
Keep the laptop plugged in. It can wake itself from sleep (see the tip the script prints). Logs: `work\logs`.

### 6. Review
Double-click **`scripts\start_dashboard.bat`**. It opens http://127.0.0.1:8765.
For each video you can pick or edit the hook, title, description and hashtags, pick the thumbnail word and frame, and adjust **grade strength** with the before/after slider. Press **Re-render** after changing the hook, grade or music.

Click **Approve**, then post:
- **YouTube:** the dashboard shows the 4K file and *Copy title* / *Copy description* buttons. Studio → Create → Upload → drag it in → paste → set visibility.
- **TikTok / Instagram:** on your phone, open the package in your Drive folder → upload `video.mp4` → paste `caption.txt`. Optionally set `cover.jpg` as the cover.

---

## First time: tune the look to your references
```
.venv\Scripts\python tools\analyze_refs.py
```
It measures the 3 reference videos (contrast, black level, tint, saturation, shot length) and writes `presets\reference_look.yaml`. To use it, set `style: preset: presets/reference_look.yaml` in `config.yaml`. Compare the result in the dashboard. Every number in the preset is readable and tweakable. The rule: subtle. If it looks "edited", lower it.

## Useful commands
| Command | What it does |
|---|---|
| `python -m moto_shorts daily` | the full daily run, right now |
| `python -m moto_shorts shorts 3` | make 3 more shorts from footage you already have |
| `python -m moto_shorts long` | build a long video now |
| `python -m moto_shorts dashboard` | open the review page |
| `python -m moto_shorts check` | check setup |

(Prefix with `.venv\Scripts\` on Windows, e.g. `.venv\Scripts\python -m moto_shorts daily`.)

## Settings worth knowing (`config.yaml`)
- `daily.shorts_per_day`, `short_min_seconds` / `short_max_seconds`
- `longform.min_unused_minutes`: how much good footage must wait before a long video is built
- `style.push_in` (0 = static), `blur_sigma`, `background_dim`
- `style.music_dir`: put your own tracks here, then tick "Add music" on a video in the dashboard
- `captions.hashtag_pool`: the hashtags it rotates through (1 big + 1 mid + 1 niche per post)
- `youtube.mode`: `manual` (default) or `api`. See below.

## YouTube automatic upload (optional, later)
`manual` mode needs no Google setup. For automatic uploads, YouTube requires a Google Cloud project, and videos uploaded through an **unaudited** project are locked to private (you can't make them public). Only switch `youtube.mode: api` after the project passes YouTube's API compliance audit. Setup: Google Cloud Console → new project → enable *YouTube Data API v3* → OAuth consent screen (External, add yourself as test user) → Credentials → *OAuth client ID* → *Desktop app* → download JSON to `secrets\youtube_client_secret.json` → run `python -m moto_shorts youtube-login`.

## Privacy
Your keys live only in `.env` and `secrets\` on your computer (both are excluded from git). Footage never leaves your machine except: frames sent to Claude for caption writing, and whatever you choose to upload.
