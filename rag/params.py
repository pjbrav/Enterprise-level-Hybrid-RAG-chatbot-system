"""Per-request Hybrid tuning, driven by a single 0-1 'intensity' slider.

The engine is shared by every concurrent request, so tuning must never be
written onto it. A frozen HybridParams is passed down the call chain instead.

Only the amount of Hybrid WORK changes; Standard RAG ignores these values:

  intensity   lexical_limit  max_partitions  last_resort_threshold  last_resort_pool
  0.0         15             1               0.0 (never rerank)     10
  0.5         30             3               0.40                   20      <- engine defaults
  1.0         80             3               0.65                   40

Values between the anchors are interpolated linearly. The 0.5 row is NOT
hard-coded here: it is whatever the engine is configured with (its env-driven
defaults), so slider position 0.5 reproduces the published evaluation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

DEFAULT_INTENSITY = 0.5


@dataclass(frozen=True)
class HybridParams:
    lexical_limit: int = 30            # rows pulled per FTS / entity query (per partition)
    max_partitions: int = 3            # partitions searched separately for a question
    last_resort_threshold: float = 0.40  # rerank (cross-encoder + embeddings) when confidence is below this; 0 = never
    last_resort_pool: int = 20         # how many top candidates the rerank sees

    @classmethod
    def for_intensity(cls, intensity: float, base: "HybridParams") -> "HybridParams":
        """Params for slider position `intensity` (0-1), with `base` as the 0.5 anchor."""
        s = min(1.0, max(0.0, float(intensity)))
        if s == DEFAULT_INTENSITY:
            return base
        low = cls(lexical_limit=15, max_partitions=1, last_resort_threshold=0.0, last_resort_pool=10)
        high = cls(lexical_limit=80, max_partitions=max(3, base.max_partitions),
                   last_resort_threshold=max(0.65, base.last_resort_threshold), last_resort_pool=40)
        if s < DEFAULT_INTENSITY:
            a, b, t = low, base, s / DEFAULT_INTENSITY
        else:
            a, b, t = base, high, (s - DEFAULT_INTENSITY) / (1.0 - DEFAULT_INTENSITY)

        def mix(x, y):
            return x + (y - x) * t

        return replace(
            a,
            lexical_limit=round(mix(a.lexical_limit, b.lexical_limit)),
            max_partitions=max(1, round(mix(a.max_partitions, b.max_partitions))),
            last_resort_threshold=round(mix(a.last_resort_threshold, b.last_resort_threshold), 4),
            last_resort_pool=round(mix(a.last_resort_pool, b.last_resort_pool)),
        )
