# discord-meeting-notes

Records a Discord voice channel with one track per speaker, transcribes locally with
faster-whisper and summarises with Claude.

## Architecture

```
Discord voice  ──►  bot.py           ──►  recordings/<session>/
(Opus/RTP)          py-cord sink          <user_id>_<name>.wav   (48 kHz, stereo)
                    one track per user    session.json

recordings/<session>/  ──►  pipeline.py  ──►  transcript.md
                            faster-whisper     transcript.json
                            + Claude           summary.md
```

The bot and the pipeline are deliberately decoupled. Recording has to be stable in
real time; transcription may take as long as it takes. You can also run `pipeline.py`
on recordings from other sources (Craig, for example) as long as you have one file
per speaker.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # fill in your keys
```

GPU transcription needs the CUDA libraries:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

No GPU? Run with `--device cpu --model medium`. Expect roughly real-time speed on a
modern CPU, so about 60 minutes for a 60-minute conversation.

## The Discord app

1. https://discord.com/developers/applications → New Application → Bot
2. Copy the token into `.env`
3. Enable **Server Members Intent** under Bot → Privileged Gateway Intents
4. OAuth2 → URL Generator: scopes `bot` + `applications.commands`,
   permissions: Connect, Speak, Change Nickname, Send Messages
5. Invite the bot with the generated link

## Running

```powershell
python bot.py
```

In Discord: `/record` → the meeting → `/stop`. The bot saves the tracks, runs
transcription and summarisation automatically, and posts the result in the text
channel with `summary.md` and `transcript.md` attached.

`/summary` re-runs the pipeline on the latest session, or on a given directory:
`/summary session:2026-08-31_19-04-12`.

The pipeline can also be run standalone, for example on recordings from Craig:

```powershell
python pipeline.py recordings\2026-08-31_19-04-12
python pipeline.py recordings\2026-08-31_19-04-12 --device cpu --model medium
```

### Silence watchdog

If no audio arrives from anyone for 3 minutes while at least 2 people are in the
voice channel, the bot sends an extra UDP keepalive. If that does not help within
30 s it warns in the text channel, and says so when the audio is back. The gaps are
recorded in `session.json` and shown at the top of `summary.md`. The thresholds are
set in `.env` with `SILENCE_WARN_MIN` and `SILENCE_WARN_MIN_HUMANS`.

Timestamped logs end up in `bot.log`.

## Known limitations

- **py-cord from a PR branch.** Voice receiving only works on
  `fix/voice-rec-2` (see `requirements.txt`), and `bot.py` patches two bugs in it.
  Otherwise a single malformed Opus packet kills the entire recording. `UDPKeepAlive`
  sends a keepalive every 83 minutes instead of every 5 seconds, so Discord stops
  sending audio after 5–9 minutes. The log says when the keepalive patch is no
  longer needed.
- **Track sync.** Each track is padded with silence against the wall clock, so all
  tracks share a timeline and are the same length. Time is measured when the packet
  is processed rather than when the word was said, so the ordering between speakers
  can be off by up to the length of a thread stall, typically under a second.
- **6 h+ sessions** should be chunked before summarisation. A 2-hour transcript is
  around 30k tokens and fits fine in one call.
- Whisper hallucinates on silent audio. `vad_filter=True` is therefore not optional.

---

*Note: the code comments in this repository are written in Swedish.*
