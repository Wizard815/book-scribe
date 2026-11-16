"""Tiered failover. A call goes to the first healthy provider in tier order; later tiers are only reached
when every earlier provider failed (or is cooling down after a recent failure).

Failure = anything that stops the call producing a usable answer: connection error, timeout, HTTP error
(5xx *and* 4xx, since a wrong key or model name on T1 should not stop the writer), or an unparseable reply.
After a failure the provider is skipped for `cooldown` seconds so a dead GPU box costs one timeout, not one per
request. Skipped providers are still tried as a last resort if everything else failed.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from providers import Profile, Registry

T = TypeVar("T")


class NotConfigured(Exception):
    pass


class AllTiersFailed(Exception):
    def __init__(self, role: str, errors: list[str]):
        self.role, self.errors = role, errors
        super().__init__(f"Every {role} provider failed: " + " | ".join(errors))


@dataclass
class Health:
    ok: float = 0.0
    failed: float = 0.0
    error: str = ""
    fails_in_row: int = 0
    calls: int = 0


@dataclass
class Served:
    role: str
    tier: int  # 0-based
    label: str
    profile: str
    fell_back: bool


@dataclass
class Router:
    reg: Registry
    now: Callable[[], float] = time.monotonic
    health: dict[str, Health] = field(default_factory=dict)
    last: dict[str, Served] = field(default_factory=dict)
    prefer: dict[str, str] = field(default_factory=dict)  # role -> "auto" | "local" | "cloud" (the writer's choice; see _ordered)

    def _cooling(self, p: Profile) -> bool:
        h = self.health.get(p.id)
        return bool(h and h.fails_in_row and self.now() - h.failed < self.reg.cooldown)

    async def call(self, role: str, fn: Callable[[Profile], Awaitable[T]]) -> T:
        chain = self.reg.chain(role)
        if not chain:
            raise NotConfigured(f"No {role} provider is in any tier. Open the admin menu and assign one.")
        if role == "embed":  # vector spaces differ between models; mixing them would corrupt memory
            chain = chain[:1]
        chain = self._ordered(role, chain)
        ready = [c for c in chain if not self._cooling(c[1])]
        order = ready + [c for c in chain if c not in ready]
        errors: list[str] = []
        for tier, p in order:
            h = self.health.setdefault(p.id, Health())
            h.calls += 1
            try:
                result = await fn(p)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                h.failed, h.fails_in_row, h.error = self.now(), h.fails_in_row + 1, f"{e.__class__.__name__}: {str(e)[:160]}"
                errors.append(f"{self.reg.labels[tier]} / {p.label}: {h.error}")
                continue
            h.ok, h.fails_in_row, h.error = self.now(), 0, ""
            self.last[role] = Served(role, tier, self.reg.labels[tier], p.id, fell_back=tier != chain[0][0] or bool(errors))
            return result
        raise AllTiersFailed(role, errors)

    def _ordered(self, role: str, chain: list) -> list:
        """The writer's local/cloud choice. The last tier is the cloud (frontier) tier; every tier before it runs on the writer's own machines.
        local: never leave the machines (the cloud tier is not used at all, so nothing is sent to a third party).
        cloud: the cloud tier first, local ones as the fallback. auto: tier order, cloud only when the local tiers failed."""
        mode = self.prefer.get(role, "auto")
        if role == "embed" or mode == "auto" or len(self.reg.labels) < 2:
            return chain
        cloud_tier = len(self.reg.labels) - 1
        local = [c for c in chain if c[0] != cloud_tier]
        cloud = [c for c in chain if c[0] == cloud_tier]
        if mode == "local":
            if not local:
                raise NotConfigured(f"You chose local {role}, but no local {role} provider is set up. Ask the administrator, or choose Automatic.")
            return local
        return cloud + local if cloud else chain

    def health_snapshot(self) -> dict:
        t = self.now()
        return {pid: {"ok_ago": round(t - h.ok) if h.ok else None, "failed_ago": round(t - h.failed) if h.failed else None,
                      "error": h.error, "cooling": bool(h.fails_in_row and t - h.failed < self.reg.cooldown), "calls": h.calls}
                for pid, h in self.health.items()}

    def served(self, *roles: str) -> dict:
        return {r: {"tier": s.tier + 1, "label": s.label, "profile": s.profile, "fell_back": s.fell_back}
                for r in roles if (s := self.last.get(r))}
