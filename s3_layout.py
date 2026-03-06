"""
S3 layout helper for mmWave dataset.

New canonical layout (recommended):
  <root_prefix>/<user>/<user>-YYYYMMDD/<session_stem>/<session_stem>.(csv|mp4|wav|txt)
  <root_prefix>/<user>/<user>-YYYYMMDD/voice_drift/*.wav and _DONE.json

Backward-compatible readers should also look for legacy layouts:
  <root_prefix>/<user>-YYYYMMDD/<session_stem>/...
  <root_prefix>/<user>_YYYYMMDD/<session_stem>/...   (if present)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class S3Layout:
    root_prefix: str = ""  # e.g. "mmwave-raw-data/" or ""

    def _join(self, *parts: str) -> str:
        clean = []
        for p in parts:
            if not p:
                continue
            p = str(p).strip("/").replace("\\", "/")
            if p:
                clean.append(p)
        if not clean:
            return ""
        return "/".join(clean) + "/"

    def user_root(self, user: str) -> str:
        return self._join(self.root_prefix, user)

    def day_dirname(self, user: str, yyyymmdd: str, sep: str = "-") -> str:
        return f"{user}{sep}{yyyymmdd}"

    # ---- canonical (writer) ----
    def canonical_day_prefix(self, user: str, yyyymmdd: str, sep: str = "-") -> str:
        return self._join(self.root_prefix, user, self.day_dirname(user, yyyymmdd, sep=sep))

    def canonical_session_prefix(self, user: str, yyyymmdd: str, session_stem: str, sep: str = "-") -> str:
        return self._join(self.root_prefix, user, self.day_dirname(user, yyyymmdd, sep=sep), session_stem)

    def canonical_voice_drift_prefix(self, user: str, yyyymmdd: str, sep: str = "-") -> str:
        return self._join(self.root_prefix, user, self.day_dirname(user, yyyymmdd, sep=sep), "voice_drift")

    # ---- legacy ----
    def legacy_day_prefixes(self, user: str, yyyymmdd: str) -> List[str]:
        # support both hyphen and underscore as day separator
        return [
            self._join(self.root_prefix, f"{user}-{yyyymmdd}"),
            self._join(self.root_prefix, f"{user}_{yyyymmdd}"),
        ]

    def legacy_voice_drift_prefixes(self, user: str, yyyymmdd: str) -> List[str]:
        return [p + "voice_drift/" for p in self.legacy_day_prefixes(user, yyyymmdd)]

    # ---- reader convenience ----
    def day_prefix_candidates(self, user: str, yyyymmdd: str) -> List[str]:
        # Prefer canonical hyphen, then canonical underscore, then legacy
        out = [
            self.canonical_day_prefix(user, yyyymmdd, sep="-"),
            self.canonical_day_prefix(user, yyyymmdd, sep="_"),
        ]
        out.extend(self.legacy_day_prefixes(user, yyyymmdd))
        # de-dup preserve order
        seen=set(); ded=[]
        for p in out:
            if p and p not in seen:
                seen.add(p); ded.append(p)
        return ded

    def voice_drift_prefix_candidates(self, user: str, yyyymmdd: str) -> List[str]:
        out = [
            self.canonical_voice_drift_prefix(user, yyyymmdd, sep="-"),
            self.canonical_voice_drift_prefix(user, yyyymmdd, sep="_"),
        ]
        out.extend(self.legacy_voice_drift_prefixes(user, yyyymmdd))
        seen=set(); ded=[]
        for p in out:
            if p and p not in seen:
                seen.add(p); ded.append(p)
        return ded
