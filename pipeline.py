"""
Transkriberar en inspelad session och gör en sammanfattning.

    python pipeline.py recordings/2026-08-31_19-04-12

Producerar i sessionsmappen:
    transcript.md   – tidsstämplad, talarmärkt utskrift
    transcript.json – samma sak strukturerat, inkl. bortfiltrerade segment
    summary.md      – sammanfattning i standup-format
"""

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv

# Boten skickar med sin egen miljö till subprocessen, men pipeline.py körs
# också fristående (README visar det för Craig-inspelningar). Utan detta
# hittas inte ANTHROPIC_API_KEY och sammanfattningen hoppas tyst över.
load_dotenv(Path(__file__).parent / ".env")


def _preload_cuda_libs() -> None:
    """Gör pip-installerade CUDA-bibliotek synliga för ctranslate2.

    ctranslate2 dlopen:ar libcublas/libcudnn på soname, men nvidia-*-hjulen
    lägger dem i site-packages, som inte ligger på den dynamiska loaderns
    sökväg. Laddar vi dem först på full sökväg återanvänder en senare
    dlopen samma handtag. Alternativet är att kräva LD_LIBRARY_PATH i varje
    skal som startar boten, vilket är lätt att glömma.

    Flera pass: biblioteken beror på varandra (cublas -> cublasLt,
    cudnn -> cudnn_graph), och ett som misslyckas i ett pass kan lyckas i
    nästa när dess beroende väl är inne.
    """
    import ctypes
    import importlib.util

    sos: list[str] = []
    for pkg in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_nvrtc.lib"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        d = Path(list(spec.submodule_search_locations)[0])
        sos.extend(str(f) for f in sorted(d.glob("*.so*")))

    remaining = sos
    while remaining:
        failed = []
        for so in remaining:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(so)
        if len(failed) == len(remaining):
            break  # inget mer går att lösa upp
        remaining = failed


_preload_cuda_libs()

from faster_whisper import WhisperModel

# Domänord som taligenkänningen annars mal sönder: "BM25" har blivit "BN25",
# "Reciprocal Rank Fusion" har blivit "Reki Procol Rank Fusion" och "Jira"
# har blivit "giran". Sammanfattningen tar sådana fel på allvar och bygger
# vidare på dem, så de kostar mer än de ser ut att göra.
HOTWORDS = (
    "BM25, Reciprocal Rank Fusion, RRF, hybridsökning, semantisk sökning, "
    "metadatafiltrering, retrieval, chunk, chunking, embedding, embeddings, "
    "pgvector, Postgres, Pagila, RAG, Docker, Jira, sprint, backlog, ticket, "
    "story, deluppgift, ADR, konformans, top_k, OpenAI, Claude, "
    "harness, jämförelseharness, testkörare, demo, "
    "Elias, Emil, Bella, Sonia, Gabriella, Lullo"
)

SYSTEM_PROMPT = """Du får en talarmärkt utskrift från en scrum-liknande standup
på Discord med några få deltagare. Utskriften kommer från automatisk
taligenkänning: den innehåller fel, avbrutna meningar, och eftersom folk pratar
i mun på varandra ligger repliker ofta interfolierade. Tolka välvilligt men
hitta inte på innehåll.

Skriv på svenska, i markdown, med exakt dessa rubriker:

## Sammanfattning
3–5 meningar om vad mötet handlade om.

## Per person
En underrubrik per deltagare som sa något av substans. Under varje:
- **Gjort:** vad de rapporterade som klart eller pågående
- **Ska göra:** vad de sa att de tar härnäst
- **Blockers:** vad som hindrar dem, eller "Inga nämnda"

Utelämna helt personer som bara gav korta inpass utan innehåll.

## Beslut
Punktlista över saker gruppen faktiskt bestämde. Skriv "Inga tydliga beslut."
om det inte finns några.

## Action items
Punktlista: vad som ska göras, vem som äger det, deadline om den nämndes.
Skriv "Inga." om det inte finns några.

## Öppna frågor
Saker som lyftes men lämnades ohanterade.

Om något är otydligt i utskriften, markera det med "(osäkert)" istället för att
gissa. Om samma sak verkar sagd av två personer beror det troligen på
mikrofonläckage – tillskriv den då till en person och markera "(osäker talare)".
"""

# Vanliga backchannels. Ensamma är de skräp; i en längre mening är de fine.
BACKCHANNELS = {
    "ja", "nej", "mm", "mhm", "aa", "okej", "ok", "precis", "japp", "jo",
    "just det", "absolut", "exakt", "yes", "yeah", "tack", "hej", "hallå",
}

# Fraser Whisper hittar på ur tystnad och brus, inlärda från undertextade
# videor. De står i korta segment, ofta med en påhittad följdmening ("Tack för
# att ni har tittat på den här videon. Vi hörs om det är något."), så de fångas
# bara upp till en viss längd. I en lång replik är frasen troligen riktigt tal.
HALLUCINATIONS = (
    "tack för att du tittade", "tack för att ni tittade",
    "tack för att du har tittat", "tack för att ni har tittat",
    "den här videon", "undertexter av", "prenumerera",
)
HALLUCINATION_MAX_WORDS = 20


def format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def normalize(text: str) -> str:
    return re.sub(r"[^\wåäö ]", "", text.lower()).strip()


def is_junk(seg_text: str, duration: float, no_speech_prob: float, avg_logprob: float):
    """Returnerar en anledning om segmentet ska bort, annars None.

    Fyra oberoende signaler, för de fångar olika fel:
      no_speech_prob – Whisper tror själv att det inte var tal
      avg_logprob    – låg konfidens, typiskt hallucination eller mumlande
      hallucination  – känd påhittad fras, ofta med hög konfidens
      backchannel    – korrekt transkriberat men innehållslöst
    """
    norm = normalize(seg_text)
    if no_speech_prob > 0.6:
        return "no_speech"
    if avg_logprob < -1.0:
        return "low_confidence"
    if len(norm.split()) <= HALLUCINATION_MAX_WORDS and any(
        phrase in norm for phrase in HALLUCINATIONS
    ):
        return "hallucination"
    if norm in BACKCHANNELS and duration < 2.0:
        return "backchannel"
    if len(norm) < 2:
        return "too_short"
    return None


def drop_mic_bleed(events, similarity: float = 0.75, window_s: float = 2.0):
    """Tar bort repliker som är samma text från två talare vid samma tid.

    Uppstår när någon kör högtalare istället för hörlurar: deras mikrofon
    plockar upp de andra. Vi behåller den med högst konfidens – den som
    faktiskt pratade sitter närmast sin mikrofon.
    """
    kept, removed = [], []
    for e in events:
        dup_idx = None
        for i, k in enumerate(kept):
            if k["speaker"] == e["speaker"]:
                continue
            if abs(k["start"] - e["start"]) > window_s:
                continue
            ratio = SequenceMatcher(
                None, normalize(k["text"]), normalize(e["text"])
            ).ratio()
            if ratio >= similarity:
                dup_idx = i
                break

        if dup_idx is None:
            kept.append(e)
        elif e["logprob"] > kept[dup_idx]["logprob"]:
            loser = kept[dup_idx]
            loser["dropped"] = "mic_bleed"
            removed.append(loser)
            kept[dup_idx] = e
        else:
            e["dropped"] = "mic_bleed"
            removed.append(e)
    return kept, removed


def merge_consecutive(events, max_gap_s: float = 2.0):
    """Slår ihop repliker från samma talare som ligger tätt. Avbryter någon
    annan emellan bryts sammanslagningen – det speglar samtalet korrekt."""
    merged = []
    for e in events:
        if (
            merged
            and merged[-1]["speaker"] == e["speaker"]
            and e["start"] - merged[-1]["end"] <= max_gap_s
        ):
            merged[-1]["text"] += " " + e["text"]
            merged[-1]["end"] = e["end"]
        else:
            merged.append(dict(e))
    return merged


def transcribe_session(session_dir: Path, model_size: str, language: str, device: str):
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Laddar {model_size} på {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    events, dropped = [], []
    for wav in sorted(session_dir.glob("*.wav")):
        speaker = wav.stem.split("_", 1)[-1].replace("-", " ")
        print(f"  transkriberar {wav.name} ...")

        segments, _info = model.transcribe(
            str(wav),
            language=language,
            # VAD är kritiskt: varje spår är mest tystnad, och Whisper
            # hallucinerar gärna text ur tystnad. VAD klipper bort den men
            # behåller ursprungliga tidsstämplar.
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            # Utan detta kan modellen fastna i upprepningsloopar.
            condition_on_previous_text=False,
            beam_size=5,
            # hotwords biasar varje fönster. Ingen initial_prompt: den gällde
            # bara första fönstret, och när spåret börjar tyst skriver Whisper
            # ut prompten själv som en replik ("Deltagarna pratar om
            # sökstrategier...", tillskriven den som äger spåret).
            hotwords=HOTWORDS,
        )
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            item = {
                "start": seg.start,
                "end": seg.end,
                "speaker": speaker,
                "text": text,
                "logprob": round(seg.avg_logprob, 3),
                "no_speech": round(seg.no_speech_prob, 3),
            }
            reason = is_junk(text, seg.end - seg.start, seg.no_speech_prob, seg.avg_logprob)
            if reason:
                item["dropped"] = reason
                dropped.append(item)
            else:
                events.append(item)

    events.sort(key=lambda e: e["start"])
    events, bleed = drop_mic_bleed(events)
    dropped.extend(bleed)
    print(
        f"  {len(events)} repliker behållna, {len(dropped)} bortfiltrerade "
        f"({len(bleed)} som mikrofonläckage)"
    )
    return merge_consecutive(events), dropped


def render_transcript(events) -> str:
    return "\n\n".join(
        f"[{format_ts(e['start'])}] **{e['speaker']}:** {e['text']}" for e in events
    )


def load_gaps(session_dir: Path) -> list[dict]:
    """Perioder då boten inte tog emot ljud från någon, enligt session.json.

    Tomt för inspelningar från andra källor, t.ex. Craig, som saknar filen.
    """
    path = session_dir / "session.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("gaps", [])


def format_gaps(gaps) -> str:
    return ", ".join(f"{format_ts(g['start_s'])}–{format_ts(g['end_s'])}" for g in gaps)


def summarize(transcript: str, gaps=(), model: str = "claude-sonnet-5") -> str:
    import anthropic

    content = transcript
    if gaps:
        # Utan detta skriver modellen en sammanfattning som ser komplett ut,
        # och en lucka på en timme läses som att inget mer sades.
        content = (
            f"OBS: inspelningen saknar ljud under {format_gaps(gaps)} "
            "(mottagningen föll bort, mötet fortsatte). Säg i sammanfattningen "
            "att den bara täcker resten, och påstå inte att något inte togs upp.\n\n"
            + transcript
        )

    client = anthropic.Anthropic()  # läser ANTHROPIC_API_KEY från env
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session_dir", type=Path)
    p.add_argument("--model", default="large-v3", help="t.ex. large-v3, turbo, medium")
    p.add_argument("--language", default="sv")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-summary", action="store_true")
    args = p.parse_args()

    session_dir = args.session_dir
    transcript_path = session_dir / "transcript.md"

    if transcript_path.exists():
        print(f"Återanvänder befintlig {transcript_path}")
        transcript = transcript_path.read_text(encoding="utf-8")
    else:
        events, dropped = transcribe_session(
            session_dir, args.model, args.language, args.device
        )
        transcript = render_transcript(events)
        transcript_path.write_text(transcript, encoding="utf-8")
        (session_dir / "transcript.json").write_text(
            json.dumps({"events": events, "dropped": dropped}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Skrev {transcript_path} ({len(events)} repliker)")

    if args.no_summary:
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY saknas – hoppar över sammanfattning.")
        return

    print("Sammanfattar...")
    gaps = load_gaps(session_dir)
    summary = summarize(transcript, gaps)
    if gaps:
        # Överst, så att den syns i Discord innan någon läser vidare.
        summary = (
            f"> ⚠️ Inspelningen saknar ljud {format_gaps(gaps)}. "
            "Sammanfattningen täcker bara resten av mötet.\n\n" + summary
        )
    (session_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Skrev {session_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
