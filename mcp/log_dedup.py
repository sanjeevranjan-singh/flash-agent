"""
Greedy log deduplication — adapted from microsoft/AIOpsLab.

Collapses repeated log lines that differ only in timestamp values, freeing
token budget for more recent / more-relevant evidence in the LLM prompt.

The compressor recognises common timestamp shapes (ISO-8601, RFC3339,
HH:MM:SS, k8s durations like 5m30s, syslog) and treats two consecutive
blocks of `block_size` lines as equivalent if the only difference is
timestamp content.  A multi-pass driver runs block_size = 1 .. window_size
so identical 1-line repeats, identical 2-line pairs, etc. all collapse.

Controlled by env `LOG_TRIM` (window_size, default 0 = disabled).
Set LOG_TRIM=8 in production for ~30-50%% log byte reduction with no loss
of distinct-event signal.

Reference: microsoft/AIOpsLab :: aiopslab/orchestrator/actions/log_deduplication.py
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Pattern, Tuple


_DEFAULT_TIMESTAMP_REGEX = (
    r"(?:"
    # ISO / RFC3339 with optional ms + offset
    r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"
    r"|"
    # Abbreviated month
    r"\d{4}-[A-Z][a-z]{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"|"
    # HH:MM:SS standalone
    r"\b\d{2}:\d{2}:\d{2}\b"
    r"|"
    # k8s durations
    r"\b\d+m(?:\d+s)?\b"
    r"|"
    r"\b\d+s\b"
    r"|"
    # syslog
    r"[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4}"
    r")"
)

_DEFAULT_TS_RX: Pattern[str] = re.compile(_DEFAULT_TIMESTAMP_REGEX)


def _find_timestamp_spans(line: str, ts_rx: Pattern[str]) -> List[Tuple[int, int]]:
    return [m.span() for m in ts_rx.finditer(line)]


def _make_blocks(lines: List[str], block_size: int) -> List[str]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return ["\n".join(lines[i:i + block_size]) for i in range(0, len(lines), block_size)]


def _mask_timestamps(text: str, spans: List[Tuple[int, int]]) -> str:
    parts: List[str] = []
    last_end = 0
    for start, end in spans:
        parts.append(text[last_end:start])
        parts.append(" " * (end - start))
        last_end = end
    parts.append(text[last_end:])
    return "".join(parts)


def _greedy_compress_pass(
    lines: List[str],
    ts_rx: Pattern[str],
    block_size: int,
) -> List[str]:
    if not lines:
        return []
    blocks = _make_blocks(lines, block_size)
    result: List[str] = [blocks[0]]
    prev_spans: Optional[List[Tuple[int, int]]] = _find_timestamp_spans(blocks[0], ts_rx)

    for block in blocks[1:]:
        spans = _find_timestamp_spans(block, ts_rx)
        if not prev_spans or not spans:
            result.append(block)
            prev_spans = spans
            continue
        if len(spans) != len(prev_spans):
            result.append(block)
            prev_spans = spans
            continue
        if [s[0] for s in spans] != [s[0] for s in prev_spans]:
            result.append(block)
            prev_spans = spans
            continue
        prev_masked = _mask_timestamps(result[-1], prev_spans)
        curr_masked = _mask_timestamps(block, spans)
        if prev_masked == curr_masked:
            # collapse: keep the most recent timestamp instance
            result[-1] = block
        else:
            result.append(block)
        prev_spans = spans
    return result


def greedy_compress_lines(
    raw: str,
    window_size: Optional[int] = None,
    ts_rx: Pattern[str] = _DEFAULT_TS_RX,
) -> str:
    """Multi-pass greedy timestamp dedup.

    `window_size` defaults to `LOG_TRIM` env var (int).  A value <= 0 disables
    compression (returns input unchanged) so the feature is safe-by-default.
    """
    if window_size is None:
        try:
            window_size = int(os.environ.get("LOG_TRIM", "0"))
        except ValueError:
            window_size = 0
    if not window_size or window_size <= 0 or not raw:
        return raw
    lines = raw.splitlines()
    if not lines:
        return raw
    result = lines[:]
    for size in range(1, window_size + 1):
        result = _greedy_compress_pass(result, ts_rx, size)
    return "\n".join(result)
