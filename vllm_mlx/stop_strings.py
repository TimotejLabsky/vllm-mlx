"""Stop-string enforcement for the batched path (PATCHES.md #32).

mlx-lm's BatchGenerator only understands stop *token ids*; the batched
schedulers never read ``sampling_params.stop``, so API/parser stop strings
were silently unenforced under ``--continuous-batching``. These helpers scan
generated text at the engine layer.

Scan discipline mirrors patch #17's SimpleEngine hygiene: a bounded tail of
``max(len(stop))-1`` characters carries across chunks, so each chunk costs
O(len(new_text)), never a rescan of the accumulated stream. Unlike
SimpleEngine (which finishes at chunk granularity and can leak the stop text
into the final chunk), the scanner cuts the emitted text at the earliest
match start — OpenAI semantics: the stop sequence is not returned.
"""


def _earliest_match(text: str, stop_list: list[str]) -> int:
    cut = -1
    for s in stop_list:
        i = text.find(s)
        if i != -1 and (cut == -1 or i < cut):
            cut = i
    return cut


def truncate_at_stop(text: str, stop) -> tuple[str, bool]:
    """Cut a COMPLETE text at the earliest stop-string match.

    Returns (text, hit). For the non-stream path.
    """
    stop_list = [s for s in (stop or []) if s]
    if not stop_list or not text:
        return text, False
    cut = _earliest_match(text, stop_list)
    if cut == -1:
        return text, False
    return text[:cut], True


class StopStringScanner:
    """Incremental scanner for the streaming path.

    ``scan(new_text)`` returns ``(emit_text, hit)``: on a hit, ``emit_text``
    is the part of ``new_text`` before the earliest match (possibly ``""``
    when the match started inside the carried tail — that prefix was already
    emitted in an earlier chunk and cannot be unsent; stop markers are
    normally single tokens, so this is the rare case).
    """

    def __init__(self, stop):
        self.stop_list = [s for s in (stop or []) if s]
        self.max_len = max((len(s) for s in self.stop_list), default=0)
        self._tail = ""

    @property
    def active(self) -> bool:
        return self.max_len > 0

    def _advance(self, combined: str) -> None:
        self._tail = combined[-(self.max_len - 1) :] if self.max_len > 1 else ""

    def scan(self, new_text: str) -> tuple[str, bool]:
        if not self.active or not new_text:
            return new_text, False
        combined = self._tail + new_text
        cut = _earliest_match(combined, self.stop_list)
        if cut == -1:
            self._advance(combined)
            return new_text, False
        emit_upto = cut - len(self._tail)
        return (new_text[:emit_upto] if emit_upto > 0 else ""), True

    def deferred_scan(self, new_text: str) -> bool:
        """Consume a chunk WITHOUT stopping; report whether a match was seen.

        Used while a structured-output grammar is mid-value (PATCHES.md #89):
        the stop must not fire, but the carried tail still has to advance so
        that later chunks scan correctly, and a suppressed match is worth
        counting. Because the tail is only ``max_len - 1`` characters, a
        given match is reported once, never re-reported on the next chunk.
        """
        if not self.active or not new_text:
            return False
        combined = self._tail + new_text
        cut = _earliest_match(combined, self.stop_list)
        self._advance(combined)
        return cut != -1
