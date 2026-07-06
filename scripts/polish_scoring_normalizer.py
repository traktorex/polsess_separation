"""Scoring-time Polish text normalizer — SECONDARY metric only.

Implements the four rules that survived the 2026-07-02 normalization-fairness
census (118 test fragments swept by 20 readers for same-content/different-form
clashes; 5 rules synthesized; 3-lens adversarial panel — false-positives,
Polish morphology, test-adaptation — kept these 4 at 3/3 votes each). Full
provenance: ~/datasets/eval/clarin_fragments/_forensics/norm_census/NORM_CENSUS_REPORT.md.

Applied SYMMETRICALLY to reference and hypothesis before scoring. The
pre-registered primary numbers stay unnormalized; normalized WER/CER is
reported alongside, never instead.

Rules (application order matters, see normalize_text):
  1. ABBR-DISCOURSE   np./itd./itp./m.in./tzn./tj. -> spoken expansion
                      (invariant discourse markers only; unit/currency
                      abbreviations excluded — their expansion inflects)
  2. INTERJ-OK        ok/okej/okay/okey/oki -> "ok"
  3. NUM-CARD-DIGIT   spelled-out cardinals -> digits, via a closed inflected
                      lexicon + conservative adjacent composition (only chains
                      of strictly descending magnitude, scale words compose);
                      "raz"/"razy" deliberately absent (means once/times);
                      unknown forms pass through untouched (safe failure)
  4. ORTHO-HYPHEN-APOS intra-word hyphen -> space; apostrophe at an
                      alphanumeric boundary deleted ("Total War'a" -> "Total Wara")

Rejected by the panel (0/3): ORD-DIGIT ordinal collapse — mapping ordinals to
bare digits collides with cardinals ("drugi"->2 == "dwa"->2) and would forgive
genuine ASR errors.
"""
from __future__ import annotations

import re

# --- rule 1: invariant discourse abbreviations -> spoken form ---------------
_ABBR = [
    (re.compile(r"(?i)(?<![\w.])m\.in\."), "między innymi"),
    (re.compile(r"(?i)(?<![\w.])np\."), "na przykład"),
    (re.compile(r"(?i)(?<![\w.])itd\."), "i tak dalej"),
    (re.compile(r"(?i)(?<![\w.])itp\."), "i tym podobne"),
    (re.compile(r"(?i)(?<![\w.])tzn\."), "to znaczy"),
    (re.compile(r"(?i)(?<![\w.])tj\."), "to jest"),
]

# --- rule 2: the ok-interjection spelling set -> "ok" ------------------------
_OK = re.compile(r"(?i)(?<!\w)(?:okej|okay|okey|oki|ok)(?!\w)")

# --- rule 3: closed lexicon of inflected Polish cardinals --------------------
# class 'u' = units/teens (<20), 't' = tens, 'h' = hundreds, 's' = scales.
_CARD: dict[str, tuple[int, str]] = {}
for _val, _cls, _forms in [
    (0, "u", "zero"),
    (1, "u", "jeden jedna jedno jednego jednej jednym jedną"),
    (2, "u", "dwa dwie dwóch dwom dwoma dwiema dwu"),
    (3, "u", "trzy trzech trzem trzema"),
    (4, "u", "cztery czterech czterem czterema"),
    (5, "u", "pięć pięciu pięcioma"),
    (6, "u", "sześć sześciu sześcioma"),
    (7, "u", "siedem siedmiu siedmioma"),
    (8, "u", "osiem ośmiu ośmioma"),
    (9, "u", "dziewięć dziewięciu dziewięcioma"),
    (10, "u", "dziesięć dziesięciu dziesięcioma"),
    (11, "u", "jedenaście jedenastu"),
    (12, "u", "dwanaście dwunastu dwunastoma"),
    (13, "u", "trzynaście trzynastu"),
    (14, "u", "czternaście czternastu"),
    (15, "u", "piętnaście piętnastu"),
    (16, "u", "szesnaście szesnastu"),
    (17, "u", "siedemnaście siedemnastu"),
    (18, "u", "osiemnaście osiemnastu"),
    (19, "u", "dziewiętnaście dziewiętnastu"),
    (20, "t", "dwadzieścia dwudziestu dwudziestoma"),
    (30, "t", "trzydzieści trzydziestu"),
    (40, "t", "czterdzieści czterdziestu"),
    (50, "t", "pięćdziesiąt pięćdziesięciu"),
    (60, "t", "sześćdziesiąt sześćdziesięciu"),
    (70, "t", "siedemdziesiąt siedemdziesięciu"),
    (80, "t", "osiemdziesiąt osiemdziesięciu"),
    (90, "t", "dziewięćdziesiąt dziewięćdziesięciu"),
    (100, "h", "sto stu"),
    (200, "h", "dwieście dwustu"),
    (300, "h", "trzysta trzystu"),
    (400, "h", "czterysta czterystu"),
    (500, "h", "pięćset pięciuset"),
    (600, "h", "sześćset sześciuset"),
    (700, "h", "siedemset siedmiuset"),
    (800, "h", "osiemset ośmiuset"),
    (900, "h", "dziewięćset dziewięciuset"),
    (1_000, "s", "tysiąc tysiące tysięcy"),
    (1_000_000, "s", "milion miliony milionów"),
]:
    for _f in _forms.split():
        _CARD[_f] = (_val, _cls)

_TOKEN = re.compile(r"^(\W*)([\w]+)(\W*)$", re.UNICODE)
_RANK = {"u": 1, "t": 2, "h": 3}


def _digits_pass(text: str) -> str:
    """Rewrite maximal runs of adjacent cardinal words to one digit token."""
    toks = text.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        m = _TOKEN.match(toks[i])
        core = m.group(2).lower() if m else None
        if core not in _CARD:
            out.append(toks[i])
            i += 1
            continue
        # Start a number group at token i; extend conservatively.
        prefix = m.group(1)
        total, cur, last_rank = 0, 0, 99   # rank of last sub-scale word
        j = i
        while j < len(toks):
            mj = _TOKEN.match(toks[j])
            if not mj or mj.group(2).lower() not in _CARD:
                break
            val, cls = _CARD[mj.group(2).lower()]
            if cls == "s":
                if total and cur == 0:      # two scale words in a row etc.
                    break
                total += max(cur, 1) * val
                cur, last_rank = 0, 99
            else:
                if _RANK[cls] >= last_rank:  # not strictly descending -> new number
                    break
                cur += val
                last_rank = _RANK[cls]
            j += 1
            if mj.group(3):                 # trailing punctuation ends the group
                break
            if j < len(toks):
                mn = _TOKEN.match(toks[j])
                if mn and mn.group(1):      # leading punctuation on next token
                    break
        last = _TOKEN.match(toks[j - 1])
        out.append(f"{prefix}{total + cur}{last.group(3) if last else ''}")
        i = j
    return " ".join(out)


# --- rule 4: intra-word hyphens and apostrophes ------------------------------
_HYPHEN = re.compile(r"(?<=\w)[-‐](?=\w)")
_APOS = re.compile(r"(?<=\w)['’]|['’](?=\w)")


def normalize_text(text: str) -> str:
    """Apply all four panel-approved rules to one utterance string."""
    for pat, repl in _ABBR:
        text = pat.sub(repl, text)
    text = _OK.sub("ok", text)
    text = _digits_pass(text)
    text = _HYPHEN.sub(" ", text)
    text = _APOS.sub("", text)
    return text


if __name__ == "__main__":
    # Smoke checks on the census's own examples.
    cases = [
        ("pięć godzin", "5 godzin"),
        ("dwa tysiące", "2000"),
        ("dwa tysiące dwadzieścia", "2020"),
        ("sto dwadzieścia trzy", "123"),
        ("dwadzieścia złotych", "20 złotych"),
        ("dwóch monitorów", "2 monitorów"),
        ("dwa, dwa", "2, 2"),                      # punctuation splits groups
        ("cztery cztery cztery", "4 4 4"),          # no ascending merge
        ("Wiesz np. Boxdel", "Wiesz na przykład Boxdel"),
        ("Okej, dobra.", "ok, dobra."),
        ("Total War'a", "Total Wara"),
        ("muzyk-gitarzysta", "muzyk gitarzysta"),
        ("raz na jakiś czas", "raz na jakiś czas"),  # raz NOT a numeral here
        ("Pięć.", "5."),
        ("no i tyle", "no i tyle"),                 # no numerals at all
    ]
    bad = 0
    for src, want in cases:
        got = normalize_text(src)
        status = "ok " if got == want else "FAIL"
        if got != want:
            bad += 1
        print(f"{status} {src!r} -> {got!r}" + ("" if got == want else f"  (want {want!r})"))
    raise SystemExit(1 if bad else 0)
