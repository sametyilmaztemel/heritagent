"""Exponential/linear/constant retry with backoff (G0 seed; criterion 6).

Pure timing policy: the execution loop owns the sleep hook; this class only
decides whether to retry and how long to wait.
"""

from __future__ import annotations


class RetryPolicy:
    def __init__(self, config: dict):
        self.max_retries = int(config["max_retries"])
        self.backoff = config["backoff"]
        self.base_delay_s = float(config.get("base_delay_s", 1.0))

    def should_retry(self, attempt: int) -> bool:
        """`attempt` is 1 for the first retry after the initial try."""
        return attempt <= self.max_retries

    def compute_delay(self, attempt: int) -> float:
        if self.backoff == "constant":
            return self.base_delay_s
        if self.backoff == "linear":
            return self.base_delay_s * attempt
        if self.backoff == "exponential":
            return self.base_delay_s * (2 ** (attempt - 1))
        raise ValueError(f"unknown backoff mode {self.backoff!r}")
