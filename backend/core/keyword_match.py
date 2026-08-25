# backend/core/keyword_match.py
# ─────────────────────────────────────────────────────────
# Devanagari-tolerant keyword matching over extracted document text.
#
# Keeps all "did this word appear, and where" logic in one place —
# callers say WHAT to look for, not HOW the lookup survives bad text.
#
# ── WHY THIS EXISTS ──────────────────────────────────────
# Many Maharashtra GR PDFs embed legacy non-Unicode Marathi fonts. Text
# extraction preserves the base consonants but corrupts the vowel signs
# (matras), and the corruption differs per document:
#
#   "शासन निर्णय" extracts as "शासन रनणणय"   (र substituted)
#   "शासन निर्णय" extracts as "सन खनणषय"     (ख substituted, same phrase)
#   "शिक्षण"      extracts as "रशक्षण" / "खिक्षण"
#
# Literal substring matching therefore finds almost nothing on real
# documents. Stripping the combining marks leaves the consonant skeleton,
# which survives the corruption far better, and difflib scores how close a
# window is to the target.
#
# ── EXACT AND FUZZY ARE REPORTED SEPARATELY, ON PURPOSE ──
# This module never merges, ranks, or resolves exact against fuzzy. They
# answer different questions and their scores are not comparable:
#
#   exact — literal substring. score is definitionally 1.0. Sparse but
#           certain: a hit is the real word.
#   fuzzy — mark-stripped sliding window, score is a difflib ratio in
#           [threshold, 1.0]. Catches corrupted spellings but CAN produce
#           false positives, so every hit carries the snippet it matched
#           for a human to confirm or reject.
#
# Collapsing them into one "best answer" would silently convert an
# uncertain lead into a stated fact. Callers get both and decide.
#
# Called by:
#   scripts/scan_gr_structure.py → structural scan of a GR folder
# ─────────────────────────────────────────────────────────

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional


# ── Tunables ──────────────────────────────────────────────
# Minimum difflib ratio for a fuzzy window to count as a hit.
DEFAULT_FUZZY_THRESHOLD = 0.75

# Fuzzy matching is skipped for keywords whose mark-stripped form is
# shorter than this — short skeletons match almost anything and produce
# noise. Note "प्रत" strips to 3 characters and so is exact-only at this
# default; that is deliberate, it is short enough that exact matching
# works on it.
DEFAULT_MIN_FUZZY_LEN = 4


# ── Structural section markers ────────────────────────────
# The candidate section headers of a Maharashtra GR, in no particular
# order — report/consumer order is by the position each is found at.
#
# This list lives here rather than in a caller because there are now two
# consumers with the same question ("where does this document's structure
# change?"): scripts/scan_gr_structure.py, which reports the answer, and
# core/vectorstore.py, which chunks on it. Two copies would drift, and a
# marker added for the scan but missing from the chunker would silently
# produce worse chunks with nothing to show for it.
#
# Extend by adding entries here; nothing else needs to change.
SECTION_KEYWORDS = (
    "संदर्भ",
    "वाचा",
    "प्रस्तावना",
    "शासन निर्णय",
    "शासन परिपत्रक",
    "शासन आदेश",
    "अटी व शती",
    "प्रत",
)

# The trailing distribution list ("प्रत" / "प्रति"). Structurally a marker
# like the others, but it terminates the document body rather than opening
# a section of it, so consumers treat it separately.
FOOTER_KEYWORD = "प्रत"


# Devanagari combining marks: matras, virama, nukta, anusvara/visarga/
# candrabindu, stress and accent signs, plus the zero-width joiners that
# legacy font remapping likes to sprinkle in.
_COMBINING = re.compile(
    "["
    "ऀ-ः"   # candrabindu, anusvara, visarga
    "ऺ-ॏ"   # matras + virama
    "॑-ॗ"   # stress marks, extra matras
    "ॢ-ॣ"   # vocalic l matras
    "़"          # nukta
    "‌‍"    # ZWNJ / ZWJ
    "]"
)


# ── Result types ──────────────────────────────────────────

@dataclass(frozen=True)
class ExactMatch:
    """
    A literal substring hit.

    score is always 1.0 — an exact match is certain by definition. It is
    NOT on the same scale as FuzzyMatch.score and the two must not be
    compared or ranked against each other.
    """
    keyword:   str
    position:  int            # first occurrence
    positions: tuple = ()     # every occurrence, left to right
    count:     int = 0
    snippet:   str = ""
    score:     float = 1.0

    @property
    def last_position(self) -> int:
        return self.positions[-1] if self.positions else self.position


@dataclass(frozen=True)
class FuzzyMatch:
    """
    Best mark-stripped window match for a keyword.

    Only the single best-scoring window is reported — the question is
    "is this header plausibly here, and where", not an exhaustive count.
    `snippet` is the raw text that matched, for human verification.
    """
    keyword:  str
    position: int
    score:    float           # difflib ratio, in [threshold, 1.0]
    snippet:  str


@dataclass(frozen=True)
class MatchReport:
    """
    Result of matching a keyword list against one document.

    exact / fuzzy map keyword -> match, and contain ONLY found keywords.
    exact_order / fuzzy_order list found keywords by position of
    appearance, which is what reveals document structure.
    """
    text_length: int
    exact: dict = field(default_factory=dict)   # str -> ExactMatch
    fuzzy: dict = field(default_factory=dict)   # str -> FuzzyMatch
    exact_order: tuple = ()
    fuzzy_order: tuple = ()

    def position_ratio(self, keyword: str, kind: str = "exact",
                       use_last: bool = True) -> Optional[float]:
        """
        How far into the document a keyword sits, as 0.0-1.0.
        Returns None if that keyword was not found by `kind`.

        Args:
            kind     : "exact" or "fuzzy"
            use_last : for exact, measure the LAST occurrence rather than
                       the first. A trailing section (a distribution list,
                       say) is what late-position questions are usually
                       about, and the same word often appears mid-document.
                       Ignored for fuzzy, which reports one position.
        """
        if not self.text_length:
            return None

        if kind == "exact":
            m = self.exact.get(keyword)
            if m is None:
                return None
            pos = m.last_position if use_last else m.position
        elif kind == "fuzzy":
            m = self.fuzzy.get(keyword)
            if m is None:
                return None
            pos = m.position
        else:
            raise ValueError(f"kind must be 'exact' or 'fuzzy', got {kind!r}")

        return round(pos / self.text_length, 3)

    def is_near_end(self, keyword: str, tail_pct: float = 25.0,
                    kind: str = "exact", use_last: bool = True) -> bool:
        """
        True if the keyword sits within the last `tail_pct` percent of the
        character stream. False if not found — callers wanting to tell
        "absent" from "present but early" should check position_ratio().
        """
        ratio = self.position_ratio(keyword, kind=kind, use_last=use_last)
        if ratio is None:
            return False
        return ratio >= 1.0 - (tail_pct / 100.0)


# ── Normalization ─────────────────────────────────────────

def strip_marks(text: str):
    """
    Removes Devanagari combining marks, leaving the consonant skeleton.

    Returns:
        (stripped_text, index_map) where index_map[i] is the offset in the
        ORIGINAL string that stripped_text[i] came from — so a match found
        in stripped space can be reported at its true document position.
    """
    out, index_map = [], []
    for i, ch in enumerate(text):
        if not _COMBINING.match(ch):
            out.append(ch)
            index_map.append(i)
    return "".join(out), index_map


def normalize(text: str) -> str:
    """Mark-stripped form of `text`, without the index map."""
    return strip_marks(text)[0]


# ── Matching ──────────────────────────────────────────────

def find_exact(text: str, keyword: str) -> list:
    """All start offsets of a literal substring match, left to right."""
    hits, start = [], 0
    if not keyword:
        return hits
    while True:
        i = text.find(keyword, start)
        if i == -1:
            return hits
        hits.append(i)
        start = i + 1


def find_fuzzy(text: str, keyword: str,
               threshold: float = DEFAULT_FUZZY_THRESHOLD,
               min_len: int = DEFAULT_MIN_FUZZY_LEN,
               normalized=None) -> Optional[FuzzyMatch]:
    """
    Best mark-stripped sliding-window match for `keyword`.

    Args:
        text       : raw extracted document text
        keyword    : what to look for, in correct spelling
        threshold  : minimum difflib ratio to count as a hit
        min_len    : skip keywords whose stripped form is shorter than this
        normalized : optional (stripped, index_map) from strip_marks(text),
                     to avoid re-stripping when matching many keywords
                     against the same document

    Returns:
        FuzzyMatch, or None if nothing cleared the threshold.
    """
    kw_stripped, _ = strip_marks(keyword)
    if len(kw_stripped) < min_len:
        return None

    stripped, index_map = normalized if normalized is not None else strip_marks(text)

    window = len(kw_stripped)
    best = (0.0, -1)
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(kw_stripped)

    for i in range(0, max(0, len(stripped) - window + 1)):
        matcher.set_seq1(stripped[i:i + window])
        # real_quick_ratio / quick_ratio are cheap upper bounds — skip
        # windows that cannot possibly clear the threshold.
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        score = matcher.ratio()
        if score > best[0]:
            best = (score, i)

    score, i = best
    if i == -1 or score < threshold:
        return None

    orig_pos = index_map[i]
    end = index_map[min(i + window, len(index_map)) - 1] + 1
    return FuzzyMatch(
        keyword=keyword,
        position=orig_pos,
        score=round(score, 3),
        snippet=text[orig_pos:end].replace("\n", " "),
    )


def match_keywords(text: str, keywords,
                   threshold: float = DEFAULT_FUZZY_THRESHOLD,
                   min_len: int = DEFAULT_MIN_FUZZY_LEN) -> MatchReport:
    """
    Matches every keyword against one document, exact and fuzzy, keeping
    the two sets strictly separate.

    Args:
        text      : raw extracted document text
        keywords  : iterable of keywords, in correct spelling
        threshold : minimum difflib ratio for a fuzzy hit
        min_len   : skip fuzzy matching for keywords whose stripped form
                    is shorter than this

    Returns:
        MatchReport. `exact` and `fuzzy` contain only found keywords; a
        keyword can appear in both, either, or neither, and the two are
        never reconciled into one verdict.

    Example:
        report = match_keywords(text, ["प्रस्तावना", "शासन निर्णय"])
        report.exact_order          → ('प्रस्तावना',)
        report.fuzzy['शासन निर्णय'].snippet → ' शासन रनण'
        report.is_near_end('प्रत', tail_pct=25)
    """
    report_exact, report_fuzzy = {}, {}
    exact_positions, fuzzy_positions = [], []

    if not text:
        return MatchReport(text_length=0)

    # Stripped once and reused — stripping per keyword would be O(keywords)
    # passes over the whole document for no gain.
    normalized = strip_marks(text)

    for kw in keywords:
        hits = find_exact(text, kw)
        if hits:
            report_exact[kw] = ExactMatch(
                keyword=kw,
                position=hits[0],
                positions=tuple(hits),
                count=len(hits),
                snippet=text[hits[0]:hits[0] + len(kw)].replace("\n", " "),
            )
            exact_positions.append((hits[0], kw))

        hit = find_fuzzy(text, kw, threshold=threshold, min_len=min_len,
                         normalized=normalized)
        if hit is not None:
            report_fuzzy[kw] = hit
            fuzzy_positions.append((hit.position, kw))

    exact_positions.sort()
    fuzzy_positions.sort()

    return MatchReport(
        text_length=len(text),
        exact=report_exact,
        fuzzy=report_fuzzy,
        exact_order=tuple(kw for _, kw in exact_positions),
        fuzzy_order=tuple(kw for _, kw in fuzzy_positions),
    )
