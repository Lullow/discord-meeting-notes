# discord-meeting-notes

Spelar in en Discord-röstkanal med ett spår per talare, transkriberar lokalt med
faster-whisper och sammanfattar med Claude.

## Arkitektur

```
Discord voice  ──►  bot.py           ──►  recordings/<session>/
(Opus/RTP)          py-cord sink          <user_id>_<namn>.wav   (48 kHz, stereo)
                    ett spår per user     session.json

recordings/<session>/  ──►  pipeline.py  ──►  transcript.md
                            faster-whisper     transcript.json
                            + Claude           summary.md
```

Bot och pipeline är medvetet frikopplade. Inspelningen måste vara realtidsstabil;
transkriberingen får ta den tid den tar. Du kan också köra `pipeline.py` på
inspelningar från andra källor (t.ex. Craig) så länge du har en fil per talare.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # fyll i dina nycklar
```

För GPU-transkribering behövs CUDA-biblioteken:

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

Saknar du GPU: kör med `--device cpu --model medium`. Räkna med ungefär
realtidshastighet på en modern CPU, alltså ~60 min för ett 60-minuterssamtal.

## Discord-appen

1. https://discord.com/developers/applications → New Application → Bot
2. Kopiera token till `.env`
3. Slå på **Server Members Intent** under Bot → Privileged Gateway Intents
4. OAuth2 → URL Generator: scopes `bot` + `applications.commands`,
   permissions: Connect, Speak, Change Nickname, Send Messages
5. Bjud in boten med den genererade länken

## Körning

```powershell
python bot.py
```

I Discord: `/record` → mötet → `/stop`. Boten sparar spåren, kör transkribering
och sammanfattning automatiskt, och postar resultatet i textkanalen med
`summary.md` och `transcript.md` bifogade.

`/summary` kör om pipelinen på senaste sessionen, eller på en angiven mapp:
`/summary session:2026-08-31_19-04-12`.

Pipelinen går också att köra fristående, t.ex. på inspelningar från Craig:

```powershell
python pipeline.py recordings\2026-08-31_19-04-12
python pipeline.py recordings\2026-08-31_19-04-12 --device cpu --model medium
```

### Tystnadsvakt

Kommer inget ljud från någon på 3 minuter medan minst 2 personer sitter i
röstkanalen skickar boten en extra UDP-keepalive. Hjälper inte det inom 30 s
varnar den i textkanalen och säger till när ljudet är tillbaka. Luckorna sparas
i `session.json` och visas överst i `summary.md`. Trösklarna ställs i `.env`
med `SILENCE_WARN_MIN` och `SILENCE_WARN_MIN_HUMANS`.

Loggar med tidsstämplar hamnar i `bot.log`.

## Kända begränsningar

- **py-cord från en PR-branch.** Röstmottagning fungerar bara på
  `fix/voice-rec-2` (se `requirements.txt`), och `bot.py` lappar två fel i den.
  Ett trasigt Opus-paket dödar annars hela inspelningen. `UDPKeepAlive` skickar
  keepalive var 83:e minut istället för var 5:e sekund, så Discord slutar skicka
  ljud efter 5–9 minuter. Loggen säger när keepalive-lappen inte längre behövs.
- **Spårsynk.** Varje spår fylls ut med tystnad mot väggklockan, så alla spår
  delar tidslinje och är lika långa. Tiden mäts när paketet behandlas och inte
  när ordet sades, så ordningen mellan talare kan vara fel med upp till längden
  på ett trådstopp, typiskt under en sekund.
- **6 h+ sessioner** bör chunkas innan sammanfattning. En 2-timmarsutskrift är
  runt 30k tokens och går fint i ett anrop.
- Whisper hallucinerar på tyst ljud. `vad_filter=True` är därför inte valfritt.
