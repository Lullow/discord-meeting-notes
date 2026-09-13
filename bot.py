"""
Discord-bot som spelar in en röstkanal med ett spår per talare, och därefter
kör transkribering + sammanfattning automatiskt.

Kommandon:
  /record   – joinar din röstkanal och börjar spela in
  /stop     – stoppar, sparar spåren och kör pipelinen
  /summary  – kör om pipelinen på en tidigare session
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import discord
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]
RECORDINGS_DIR = Path(os.getenv("RECORDINGS_DIR", "recordings"))
PIPELINE = Path(__file__).parent / "pipeline.py"
PIPELINE_TIMEOUT_S = 60 * 60

# Loggfil med tidsstämplar. print() buffras när utdata omdirigeras till fil, så
# ordningen där går inte att lita på, och py-cords egna INFO- och ERROR-loggar
# (voice-reconnects, DAVE-övergångar) syns inte alls utan en handler.
_log_handler = logging.FileHandler("bot.log", encoding="utf-8")
_log_handler.setLevel(logging.INFO)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
# py-cord loggar varje RTCP Sender Report på INFO, ungefär en rad i sekunden,
# och de dränker allt annat: 1097 av 1112 rader i ett 20-minuterstest.
_log_handler.addFilter(
    lambda record: "unexpected rtcp packet type=200" not in record.getMessage()
)
logging.basicConfig(level=logging.WARNING, handlers=[_log_handler])
for _name in ("discord.voice", "sound-bot"):
    logging.getLogger(_name).setLevel(logging.INFO)

_log = logging.getLogger("sound-bot")

# VOICE_DEBUG=1 loggar py-cords röstmottagning till voice-debug.log.
# DAVE-dekrypteringsfel syns bara på DEBUG-nivå. Loggen blir stor - varje
# paket ger en rad, alltså ~50 rader per sekund och talare. Bara för felsökning.
if os.getenv("VOICE_DEBUG"):
    import logging

    _h = logging.FileHandler("voice-debug.log", encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for _name in ("discord.voice", "discord.opus"):
        _lg = logging.getLogger(_name)
        _lg.setLevel(logging.DEBUG)
        _lg.addHandler(_h)


intents = discord.Intents.default()
intents.voice_states = True
intents.members = True  # behövs för att slå upp display_name

bot = discord.Bot(intents=intents)

# Räknare för paket som kastats bort av hårdningen nedan. Rapporteras efter
# varje inspelning: är den hög sker något återkommande fel i mottagningen.
_decode_errors = 0
_decode_errors_lock = threading.Lock()


def _harden_packet_router() -> None:
    """Hindrar ett korrupt Opus-paket från att döda hela inspelningen.

    `PacketRouter._do_run()` saknar felhantering runt `decoder.pop_data()`.
    Ett OpusError - typiskt "corrupted stream" efter en DAVE-rekey, alltså när
    någon går in i eller ut ur röstkanalen - propagerar ut ur mottagartråden,
    och `run()` kör då `stop_recording()` i sitt finally-block. Mottagningen
    dör tyst mitt i mötet: spåren slutar där, men boten ser ut att spela in
    vidare och `/stop` rapporterar glatt en full inspelning.

    Rapporterat i PR #3159 men inte åtgärdat. Paketet är redan hämtat ur
    jitterbufferten när felet uppstår, så det är säkert att hoppa över det -
    vi tappar 20 ms ljud istället för resten av mötet.
    """
    from discord.voice.receive.router import PacketRouter

    if getattr(PacketRouter, "_sound_bot_hardened", False):
        return

    def _do_run(self) -> None:
        global _decode_errors
        while not self._end_thread.is_set():
            self.waiter.wait()

            with self._lock:
                for decoder in self.waiter.items:
                    try:
                        data = decoder.pop_data()
                    except Exception:
                        with _decode_errors_lock:
                            _decode_errors += 1
                        _log.debug("Hoppade över trasigt paket", exc_info=True)
                        continue
                    if data is not None:
                        self.sink.write(data, data.source)

    PacketRouter._do_run = _do_run
    PacketRouter._sound_bot_hardened = True


def _patch_udp_keepalive() -> None:
    """Får UDP-keepalive att skickas var 5:e sekund istället för var 83:e minut.

    `UDPKeepAlive.delay` är 5000 och används som `time.sleep(self.delay)`,
    alltså i sekunder. En keepalive går iväg när inspelningen startar och nästa
    först efter 83:20. Discord slutar skicka ljud till en klient som inte hörts
    av på några minuter, så mottagningen dör tyst efter 5-9 minuter och kommer
    tillbaka vid 83:20. Samma kod på master och på fix/voice-rec-2.

    Villkorat: fixar upstream felet genom att byta enhet, t.ex.
    `sleep(delay / 1000)`, får vi inte sätta 5 ms. Loggen säger när lappen kan
    tas bort.
    """
    from discord.voice.receive.reader import UDPKeepAlive

    if UDPKeepAlive.delay > 60:
        _log.info("Lappar UDPKeepAlive.delay: %s -> 5 s", UDPKeepAlive.delay)
        UDPKeepAlive.delay = 5
    else:
        _log.info(
            "UDPKeepAlive.delay är %s, keepalive-lappen behövs inte längre",
            UDPKeepAlive.delay,
        )


_harden_packet_router()
_patch_udp_keepalive()


# Discord skickar 20 ms Opus-ramar.
FRAME_S = 0.02
SAMPLE_RATE = 48000
BYTES_PER_FRAME = 2 * 2  # 16-bit, stereo
BYTES_PER_S = SAMPLE_RATE * BYTES_PER_FRAME
# Så långt efter väggklockan får ett spår ligga innan tystnad fylls i. Tre
# ramar, så att vanlig nätverksjitter inte ger små hack mitt i en mening.
LAG_TOLERANCE_S = 0.06

# Tystnadsvakten: så länge utan ljud från någon, med minst så många människor
# i kanalen, innan boten reagerar. Styrbart för att kunna testa varningen ensam.
SILENCE_WARN_S = float(os.getenv("SILENCE_WARN_MIN", "3")) * 60
SILENCE_WARN_MIN_HUMANS = int(os.getenv("SILENCE_WARN_MIN_HUMANS", "2"))
# Så länge vakten väntar efter en extra keepalive innan den varnar.
SILENCE_NUDGE_WAIT_S = 30
WATCH_INTERVAL_S = 10


class TimestampedWaveSink(discord.sinks.WaveSink):
    """WaveSink som återställer den gemensamma tidslinjen.

    `sync_start` deprekerades i py-cord 2.7 och är en no-op i receive-reworken:
    spåren fylls inte längre ut med tystnad, så varje fil innehåller bara
    talarens egna paket hopklippta kant i kant. Alla spår börjar därmed på sin
    egen nolla, och ordningen mellan personer blir meningslös.

    Varje spår fylls ut mot absolut position: när ett paket kommer ska spåret
    ligga där väggklockan står. Ligger spåret efter fylls tystnad i. Ligger det
    före, efter en burst av försenade paket, väntar vi bara in klockan och
    kastar inget ljud. Felet begränsas då till en paketburst och växer inte.
    Att istället padda med tiden sedan förra paketet räknar varje jitter och
    trådstopp som tystnad utan att dra av bursten efteråt. Så drev spåren upp
    till fem minuter under ett 85-minutersmöte.

    Noggrannheten begränsas av att vi mäter när paketet *behandlades*, inte när
    ordet sades - nätverksjitter och buffring ger någon tiondels sekund.

    Räknarna används för att se var ljud försvinner: paket per minut visar när
    mottagningen dog, och `silences` perioder då ingen alls hördes.
    """

    def __init__(self, *, t0: float, filters=None):
        super().__init__(filters=filters)
        self.t0 = t0
        self.stopped_at: float | None = None
        self.stats: dict = {}
        # Skrivs i router-tråden och läses av tystnadsvakten i event-loopen.
        # Tilldelning och append är atomära under GIL:en, så inget lås behövs.
        self.last_heard: float = t0
        self.silences: list[tuple[float, float]] = []

    def _stats(self, user, elapsed: float) -> dict:
        return self.stats.setdefault(
            user,
            {
                "packets": 0,
                "first_s": round(elapsed, 2),
                "silence_s": 0.0,
                "packets_per_min": [],
            },
        )

    def _note_silence(self, now: float) -> None:
        if now - self.last_heard >= SILENCE_WARN_S:
            self.silences.append((self.last_heard - self.t0, now - self.t0))

    def _pad_to(self, user, elapsed: float, tolerance_s: float) -> None:
        audio = self.audio_data.get(user)
        pos = audio.file.tell() if audio is not None else 0
        target = int(elapsed * SAMPLE_RATE) * BYTES_PER_FRAME
        if target - pos > tolerance_s * BYTES_PER_S:
            super().write(b"\x00" * (target - pos), user)
            self._stats(user, elapsed)["silence_s"] += (target - pos) / BYTES_PER_S

    def write(self, data, user):
        now = time.monotonic()
        elapsed = now - self.t0
        self._note_silence(now)
        self.last_heard = now

        st = self._stats(user, elapsed)
        # Paketet bär de senaste 20 ms, så ljudet började en ram före nu.
        self._pad_to(user, elapsed - FRAME_S, LAG_TOLERANCE_S)
        per_min = st["packets_per_min"]
        minute = int(elapsed // 60)
        per_min.extend([0] * (minute + 1 - len(per_min)))
        per_min[minute] += 1
        st["packets"] += 1
        super().write(data, user)

    def cleanup(self):
        # Alla spår fylls ut till samma slut. Ett spår som slutar tidigt kan
        # annars misstolkas som att personen gick. Körs efter att router-tråden
        # stoppats och innan WaveSink skriver WAV-headern.
        self.stopped_at = time.monotonic()
        self._note_silence(self.stopped_at)
        elapsed = self.stopped_at - self.t0
        for user in list(self.audio_data):
            self._pad_to(user, elapsed, 0.0)
        super().cleanup()


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _humans_in(vc) -> int:
    channel = vc.channel
    return sum(1 for m in channel.members if not m.bot) if channel else 0


def _send_udp_keepalive(vc) -> None:
    """Skickar en extra UDP-keepalive vid sidan av py-cords egen tråd."""
    conn = vc._connection
    try:
        conn.socket.sendto(
            int(time.monotonic() * 1000).to_bytes(8, "big"),
            (conn.endpoint_ip, conn.voice_port),
        )
    except Exception:
        _log.warning("Kunde inte skicka extra UDP-keepalive", exc_info=True)


def find_gaps(silences, presence, min_humans: int) -> list[dict]:
    """Väljer ut de tysta perioder som är luckor i inspelningen.

    En period utan ljud är bara en lucka om folk satt i kanalen: sitter en
    person kvar tyst när mötet är slut är det ingen lucka. Majoriteten av
    närvaroproverna avgör, så att någon som hoppar ut och in inte döljer en
    timslång lucka.
    """
    gaps = []
    for start, end in silences:
        samples = [n for t, n in presence if start <= t <= end]
        present = sum(1 for n in samples if n >= min_humans)
        if samples and present * 2 >= len(samples):
            gaps.append({"start_s": round(start, 1), "end_s": round(end, 1)})
    return gaps


async def watch_silence(session: dict, sink: TimestampedWaveSink, vc) -> None:
    """Varnar i textkanalen när inget ljud kommer fram under ett möte.

    Först skickas en extra UDP-keepalive, eftersom en utebliven keepalive är
    den enda kända orsaken till att mottagningen dör. Kommer inget ljud inom
    SILENCE_NUDGE_WAIT_S varnar vi, en gång per tyst period, och säger till när
    ljudet är tillbaka. Antalet människor i kanalen sparas i session, så att
    on_recording_done kan skilja en lucka från ett avslutat möte.
    """
    below_min_at = sink.t0  # senast det satt för få i kanalen
    nudged_at = None
    warned_from = None  # last_heard när varningen gick ut

    while vc.is_recording():
        await asyncio.sleep(WATCH_INTERVAL_S)
        try:
            now = time.monotonic()
            humans = _humans_in(vc)
            session["presence"].append((round(now - sink.t0, 1), humans))
            if humans < SILENCE_WARN_MIN_HUMANS:
                below_min_at = now

            if warned_from is not None:
                if sink.last_heard > warned_from:
                    _log.info(
                        "Ljud mottaget igen efter %.0f s", sink.last_heard - warned_from
                    )
                    await session["text_channel"].send(
                        f"Ljud mottaget igen, efter "
                        f"{_fmt_ts(sink.last_heard - warned_from)} utan ljud."
                    )
                    warned_from = None
                continue

            if nudged_at is not None and sink.last_heard > nudged_at:
                _log.info("Ljudet kom tillbaka efter extra keepalive")
                nudged_at = None

            silent_s = now - max(sink.last_heard, below_min_at)
            if silent_s < SILENCE_WARN_S:
                continue

            if nudged_at is None:
                _log.warning(
                    "Inget ljud på %.0f s med %d i kanalen, skickar extra keepalive",
                    silent_s,
                    humans,
                )
                _send_udp_keepalive(vc)
                nudged_at = now
            elif now - nudged_at >= SILENCE_NUDGE_WAIT_S:
                _log.warning("Inget ljud efter extra keepalive, varnar i kanalen")
                warned_from = sink.last_heard
                nudged_at = None
                await session["text_channel"].send(
                    f"⚠️ Inget ljud mottaget på {_fmt_ts(silent_s)} fast "
                    f"{humans} personer sitter i kanalen. Mottagningen kan ha "
                    f"fallit bort – kör `/stop` och sedan `/record` om ni pratar."
                )
        except Exception:
            _log.exception("Fel i tystnadsvakten")


active = {}  # guild_id -> sessionsinfo

# Bara ett pipeline-jobb i taget. Två samtidiga large-v3 spränger VRAM på
# de flesta kort, och de skulle ändå bara slåss om samma GPU.
pipeline_lock = asyncio.Lock()


def _safe_name(name: str) -> str:
    cleaned = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    return cleaned.replace(" ", "-") or "unknown"


def split_message(text: str, limit: int = 1900):
    """Delar text i Discord-vänliga bitar vid styckesgränser."""
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 > limit:
            if current:
                chunks.append(current)
            # Ett enskilt stycke som är för långt får hårdklippas.
            while len(para) > limit:
                chunks.append(para[:limit])
                para = para[limit:]
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


async def run_pipeline(session_dir: Path, channel: discord.TextChannel):
    """Kör pipeline.py som subprocess och postar resultatet i kanalen.

    Subprocess istället för att importera pipeline direkt, av tre skäl:
      1. Whisper blockerar CPU/GPU i minuter. Kördes det i botens event loop
         skulle voice-heartbeats missas och Discord kickar boten.
      2. Modellen laddas ur helt när processen dör – ingen VRAM som ligger kvar
         mellan möten.
      3. Kraschar transkriberingen dör inte boten med den.
    """
    async with pipeline_lock:
        status = await channel.send(
            f"Transkriberar `{session_dir.name}`... det här tar några minuter."
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(PIPELINE),
            str(session_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PIPELINE.parent),
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=PIPELINE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            proc.kill()
            await status.edit(content=f"Pipelinen timade ut på `{session_dir.name}`.")
            return

        log = stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            tail = log[-1500:] if log else "(ingen output)"
            await status.edit(content=f"Pipelinen misslyckades:\n```\n{tail}\n```")
            return

    summary_path = session_dir / "summary.md"
    transcript_path = session_dir / "transcript.md"

    if not summary_path.exists():
        await status.edit(
            content=f"Utskrift klar, men ingen sammanfattning skapades.\n```\n{log[-800:]}\n```"
        )
        return

    await status.edit(content=f"**Sammanfattning – {session_dir.name}**")
    for chunk in split_message(summary_path.read_text(encoding="utf-8")):
        await channel.send(chunk)

    files = [discord.File(summary_path)]
    if transcript_path.exists():
        files.append(discord.File(transcript_path))
    await channel.send("Fullständig utskrift bifogad.", files=files)


async def on_recording_done(sink: discord.sinks.WaveSink, guild_id: int):
    """Anropas av py-cord när stop_recording() körts."""
    session = active.pop(guild_id, None)
    if session is None:
        return
    if session.get("watch_task"):
        session["watch_task"].cancel()

    session_dir: Path = session["dir"]
    session_dir.mkdir(parents=True, exist_ok=True)
    guild = bot.get_guild(guild_id)

    tracks = []
    for key, audio in sink.audio_data.items():
        # py-cord bytte nycklarna i audio_data från user-id till User/Member i
        # receive-reworken (PR #3159). Hantera båda, så koden överlever både
        # ett versionsbyte tillbaka och att fixen släpps skarpt.
        if isinstance(key, int):
            user_id = key
            member = guild.get_member(key) if guild else None
        else:
            user_id = getattr(key, "id", None)
            member = key if hasattr(key, "display_name") else None

        stats = getattr(sink, "stats", {}).get(key, {})
        display = member.display_name if member is not None else str(key)
        filename = f"{user_id}_{_safe_name(display)}.wav"
        audio.file.seek(0)
        data = audio.file.read()
        (session_dir / filename).write_bytes(data)
        tracks.append(
            {
                "user_id": str(user_id),
                "display_name": display,
                "file": filename,
                # 48 kHz, 16-bit, stereo -> 192000 byte/s. Minus 44 byte WAV-header.
                "duration_s": round(max(len(data) - 44, 0) / (48000 * 2 * 2), 2),
                # Hur mycket av spåret som är inskjuten tystnad respektive
                # mottaget ljud. Är summan mycket kortare än väggklockan
                # tappas paket någonstans i mottagningen.
                "silence_s": round(stats.get("silence_s", 0.0), 2),
                "packets": stats.get("packets", 0),
                "first_heard_s": stats.get("first_s"),
                # Visar när mottagningen dog: ett spår som går från hundratals
                # paket per minut till noll för alla samtidigt.
                "packets_per_min": stats.get("packets_per_min", []),
            }
        )

    # Sinken fyllde ut spåren till stopped_at, så väggklockan mäts dit och inte
    # till nu - on_recording_done körs en stund efter att inspelningen stoppats.
    stopped_at = getattr(sink, "stopped_at", None) or time.monotonic()
    wall_clock_s = round(stopped_at - session["t0"], 2)
    gaps = find_gaps(
        getattr(sink, "silences", []),
        session.get("presence", []),
        SILENCE_WARN_MIN_HUMANS,
    )
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "guild_id": str(guild_id),
                "started_at": session["started_at"],
                "wall_clock_s": wall_clock_s,
                # Perioder utan ljud från någon medan folk satt i kanalen.
                # pipeline.py lägger en varning om dem överst i summary.md.
                "gaps": gaps,
                "tracks": tracks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Sinken fyller ut alla spår till exakt väggklockan, och ett spår kan bara
    # ligga före med en paketburst. Är ett spår ändå tydligt längre har
    # tidsmätningen spårat ur, och tidslinjen går inte att lita på.
    overrun = [
        t["display_name"] for t in tracks if t["duration_s"] > wall_clock_s * 1.1 + 1
    ]
    # Faktiskt mottaget ljud, alltså spårlängd minus den tystnad vi själva sköt
    # in. Ligger den långt under väggklockan tappas paket i mottagningen.
    received_s = sum(t["duration_s"] - t["silence_s"] for t in tracks)

    try:
        await sink.vc.disconnect()
    except Exception:
        pass

    channel = session["text_channel"]
    andel = 100 * received_s / wall_clock_s if wall_clock_s else 0
    msg = (
        f"Inspelning klar: **{len(tracks)}** spår, {wall_clock_s / 60:.1f} min.\n"
        f"Mottaget tal: {received_s / 60:.1f} min ({andel:.0f} % av väggklockan)."
    )
    for gap in gaps:
        msg += (
            f"\n⚠️ Inget ljud mottaget {_fmt_ts(gap['start_s'])}–"
            f"{_fmt_ts(gap['end_s'])} ({(gap['end_s'] - gap['start_s']) / 60:.0f} min)."
        )
    if _decode_errors:
        msg += (
            f"\n{_decode_errors} trasiga paket hoppades över "
            f"({_decode_errors * 0.02:.1f} s ljud)."
        )
    if overrun:
        msg += (
            f"\nVarning – spår längre än inspelningen: {', '.join(overrun)}. "
            f"Tidsmätningen är opålitlig."
        )
    await channel.send(msg)

    if tracks:
        await run_pipeline(session_dir, channel)


@bot.event
async def on_ready():
    print(f"Inloggad som {bot.user} ({bot.user.id})")


@bot.slash_command(description="Börja spela in röstkanalen du sitter i")
async def record(ctx: discord.ApplicationContext):
    if ctx.guild.id in active:
        return await ctx.respond("Spelar redan in i den här servern.", ephemeral=True)

    voice = ctx.author.voice
    if not voice or not voice.channel:
        return await ctx.respond("Du sitter inte i en röstkanal.", ephemeral=True)

    # Discord stänger interaktionen om den inte kvitteras inom 3 s, och
    # voice-handshaken plus nickbytet nedan tar regelmässigt längre än så.
    # Kvittera först, skicka det riktiga svaret som followup.
    await ctx.defer()

    # En omstart mitt i en inspelning lämnar en zombie-anslutning kvar: Discord
    # tror att boten fortfarande sitter i kanalen, och connect() hänger då eller
    # kastar ClientException. Städa bort den innan vi ansluter på nytt.
    stale = ctx.guild.voice_client
    if stale is not None:
        try:
            await stale.disconnect(force=True)
        except Exception:
            pass

    # Brett except: har vi väl deferrat måste en followup skickas, annars står
    # "is thinking..." kvar i klienten i evighet och användaren får ingen ledtråd.
    try:
        vc = await voice.channel.connect(timeout=30.0)
    except Exception as e:
        return await ctx.followup.send(
            f"Kunde inte ansluta till **{voice.channel.name}**: "
            f"`{type(e).__name__}: {e}`"
        )

    started_at = datetime.now(timezone.utc).astimezone()
    session_dir = RECORDINGS_DIR / started_at.strftime("%Y-%m-%d_%H-%M-%S")

    active[ctx.guild.id] = {
        "vc": vc,
        "dir": session_dir,
        "t0": time.monotonic(),
        "started_at": started_at.isoformat(timespec="seconds"),
        "text_channel": ctx.channel,
    }

    try:
        await ctx.guild.me.edit(nick="[SPELAR IN] Notetaker")
    except discord.Forbidden:
        pass

    global _decode_errors
    with _decode_errors_lock:
        _decode_errors = 0

    session = active[ctx.guild.id]
    # Samma nolla som väggklockan i session.json.
    sink = TimestampedWaveSink(t0=session["t0"])
    try:
        vc.start_recording(
            sink,
            on_recording_done,
            ctx.guild.id,
            # sync_start används inte: den är deprekerad sedan 2.7 och en
            # no-op i receive-reworken. TimestampedWaveSink fyller luckorna.
        )
    except Exception as e:
        # Städa bort halvstartad session, annars tror /record att en inspelning
        # pågår och boten blir kvar i kanalen med inspelningsnicket kvar.
        active.pop(ctx.guild.id, None)
        try:
            await ctx.guild.me.edit(nick=None)
        except discord.Forbidden:
            pass
        await vc.disconnect(force=True)
        return await ctx.followup.send(
            f"Kunde inte starta inspelningen: `{type(e).__name__}: {e}`"
        )

    session["presence"] = []
    session["watch_task"] = asyncio.create_task(watch_silence(session, sink, vc))

    await ctx.followup.send(
        f"Spelar in **{voice.channel.name}**. Alla i kanalen spelas in – stoppa med `/stop`."
    )


@bot.slash_command(description="Stoppa inspelningen, transkribera och sammanfatta")
async def stop(ctx: discord.ApplicationContext):
    session = active.get(ctx.guild.id)
    if session is None:
        return await ctx.respond("Ingen aktiv inspelning.", ephemeral=True)

    session["text_channel"] = ctx.channel
    await ctx.respond("Stoppar och sparar...")
    session["vc"].stop_recording()  # triggar on_recording_done

    try:
        await ctx.guild.me.edit(nick=None)
    except discord.Forbidden:
        pass


@bot.slash_command(description="Kör om sammanfattningen för en tidigare session")
async def summary(
    ctx: discord.ApplicationContext,
    session: discord.Option(
        str, "Sessionsmapp, t.ex. 2026-08-31_19-04-12. Utelämna för senaste.",
        required=False,
    ) = None,
):
    sessions = sorted(d for d in RECORDINGS_DIR.glob("*") if d.is_dir())
    if not sessions:
        return await ctx.respond("Inga inspelningar hittades.", ephemeral=True)

    if session:
        session_dir = RECORDINGS_DIR / session
        if session_dir not in sessions:
            available = ", ".join(f"`{d.name}`" for d in sessions[-5:])
            return await ctx.respond(
                f"Hittar inte `{session}`. Senaste: {available}", ephemeral=True
            )
    else:
        session_dir = sessions[-1]

    # Tvinga omkörning: pipeline.py återanvänder befintlig transcript.md.
    for stale in ("summary.md",):
        (session_dir / stale).unlink(missing_ok=True)

    await ctx.respond(f"Kör pipelinen på `{session_dir.name}`.")
    await run_pipeline(session_dir, ctx.channel)


if __name__ == "__main__":
    bot.run(TOKEN)
