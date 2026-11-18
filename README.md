# Moto Shorts

Turns raw GoPro off-road POV footage into cinematic, raw-sound shorts, every day.

**What it does (once a day, when you start your computer)**
1. Cleans up: posted videos older than 7 days, and downloaded/working files older than 30 days (videos waiting for review are never deleted).
2. Downloads rides from your **whole** GoPro Cloud library, half newest and half random older ones, so old footage gets used (or picks up files you drop in `inbox`).
3. Skips anything that isn't a motorcycle ride (mountain biking etc.) and moments where no part of the bike is visible (Claude checks a few frames per video, about $0.02 each).
4. Finds the best moments by exhaust sound, color, scenery and GPS, and alternates between rides. One video can give many shorts.
5. Makes **2 shorts a day** as seamless loops, attention first: each moment is chosen so its best part (the strongest exhaust + view) opens the short. The rest follows, then a soft blend jumps back to the start at the point where sound and picture match best, so the replay joins up. Every platform gets the 9:16 blurred-background layout and a natural cinematic grade.
6. Claude writes **5 hooks**, a title, a description, 3 hashtags and **5 thumbnail words** for every video, never repeating earlier ones.
7. Builds a long 16:9 YouTube video once enough good footage has piled up.

**Dashboard:** click a hook to put it on the video (or type your own), click a thumbnail word, trim the start/end, adjust the grade, then **Approve** (moves to POSTED for 7 days, then deleted) or **Reject** (deleted, never suggested again). Use **Upload** for your own edits: a short gets blur + grade + all exports, a long video gets the grade only.

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
Install **Google Drive for desktop** (or iCloud/OneDrive), then in `config.yaml` set `handoff_dir:` to a folder inside it, e.g. `G:\My Drive\MotoShorts`. Every approved short lands there as `tiktok.mp4` + `instagram.mp4` + covers + `caption.txt`, ready on your phone.

### 5. Run it automatically when you start your computer
```
powershell -ExecutionPolicy Bypass -File scripts\install_windows_task.ps1
```
Runs 2 minutes after you sign in, at most once per day (restarting won't create extra shorts). Waking from sleep doesn't count as signing in. Logs: `work\logs`.

### 6. Review
Double-click **`scripts\start_dashboard.bat`**. It opens http://127.0.0.1:8765.
For each video you can pick or edit the hook, title, description and hashtags, pick the thumbnail word and frame, and adjust **grade strength** with the before/after slider. Press **Re-render** after changing the hook, grade or music.

Click **Approve**, then post:
- **YouTube:** the dashboard shows the 4K file and *Copy title* / *Copy description* buttons. Studio → Create → Upload → drag it in → paste → set visibility.
- **TikTok / Instagram:** on your phone, open the package in your Drive folder → upload `tiktok.mp4` (TikTok) or `instagram.mp4` (Instagram) → paste `caption.txt`. Optional covers: `cover.jpg` / `cover_instagram.jpg`.

---

## First time: tune the look to your references
```
.venv\Scripts\python tools\analyze_refs.py
```
By default it uses your 3 reference videos as the main look (75%) with 8 sampled **ScramblerPOV** videos as a light influence (25%). You can change the links and weights at the top of `tools\analyze_refs.py`. It measures them (contrast, black level, tint, saturation, shot length) and writes `presets\reference_look.yaml`. To use it, set `style: preset: presets/reference_look.yaml` in `config.yaml`. Compare the result in the dashboard. Every number in the preset is readable and tweakable. The rule: subtle. If it looks "edited", lower it.

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
