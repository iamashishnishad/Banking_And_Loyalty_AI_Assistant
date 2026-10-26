"""Per-user and per-tenant token buckets in Redis.

Both buckets are checked and charged in one Lua script, so the check is atomic across API
replicas and a request rejected by the tenant bucket doesn't use up the user's tokens. The
clock is Redis's own TIME, so replicas with skewed clocks still agree.
"""

import logging
import math
from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import get_settings

logger = logging.getLogger(__name__)

# KEYS: user bucket, tenant bucket. ARGV: user rate/s, user burst, tenant rate/s, tenant burst.
# Returns {allowed, user tokens left, seconds until allowed (string: Lua floats become ints), blocked scope}.
_TOKEN_BUCKETS = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local function refill(key, rate, burst)
  local v = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(v[1]) or burst
  local ts = tonumber(v[2]) or now
  return math.min(burst, tokens + math.max(0, now - ts) * rate)
end

local ur, ub = tonumber(ARGV[1]), tonumber(ARGV[2])
local tr, tb = tonumber(ARGV[3]), tonumber(ARGV[4])
local u = refill(KEYS[1], ur, ub)
local tn = refill(KEYS[2], tr, tb)

local allowed, wait, scope = 0, 0, ''
if u >= 1 and tn >= 1 then
  allowed = 1
  u = u - 1
  tn = tn - 1
elseif u < 1 then
  scope = 'user'
  wait = (1 - u) / ur
else
  scope = 'tenant'
  wait = (1 - tn) / tr
end

redis.call('HSET', KEYS[1], 'tokens', u, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(ub / ur) + 1)
redis.call('HSET', KEYS[2], 'tokens', tn, 'ts', now)
redis.call('EXPIRE', KEYS[2], math.ceil(tb / tr) + 1)
return {allowed, math.floor(u), tostring(wait), scope}
"""


@dataclass
class RateLimitDecision:
    allowed: bool
    remaining: int  # whole tokens left in the user's bucket
    retry_after: int  # seconds; 0 when allowed
    scope: str  # "user" | "tenant" | "" (which bucket blocked the request)


class RateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._script = redis.register_script(_TOKEN_BUCKETS)

    async def check(self, tenant: str, subject: str) -> RateLimitDecision:
        s = get_settings()
        allowed, remaining, wait, scope = await self._script(
            keys=[f"rl:user:{tenant}:{subject}", f"rl:tenant:{tenant}"],
            args=[
                s.rate_limit_user_per_minute / 60,
                s.rate_limit_user_burst,
                s.rate_limit_tenant_per_minute / 60,
                s.rate_limit_tenant_burst,
            ],
        )
        scope = scope.decode() if isinstance(scope, bytes) else scope
        return RateLimitDecision(
            allowed=bool(allowed),
            remaining=int(remaining),
            retry_after=0 if allowed else max(1, math.ceil(float(wait))),
            scope=scope,
        )
