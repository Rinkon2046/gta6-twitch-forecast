"""Forecast a big game launch day's Twitch viewers, hours 2 to 24, from its first hour.

Entry for make.fun's gta6-twitch-ccu-forecast-01 (Python 3.13, standard library only, no network).

Method: every past launch in the case's `comparables` is reduced to the shape of its release day in
log space -- for hour h, r(h) = ln(viewers_h) - ln(viewers_1). What a launch does after its first
hour depends first of all on where that first hour already sits: a game that unlocked before the
UTC day began starts hour 1 near its peak and fades, while one still waiting on a daytime unlock
starts near zero and climbs for hours. So the shapes are not pooled flat: each comparable votes
with a weight that falls off with the distance between its own first hour and this case's, in log
space (a Gaussian kernel of bandwidth 1.0). For each hour the weighted shapes give a mean and a
standard deviation, and the five numbers written are this case's hour-1 print times
exp(mean + z_tau * sd), with z_tau the standard normal's quantile at the case's levels: a
similarity-weighted, per-hour log-normal forecast of the day's shape.

A comparable whose first hour is missing or zero never votes (a category whose event began later
that day has no first hour to shape from), and an hour its source never sampled does not vote for
that hour. An hour no comparable can speak for falls back to persistence with a spread that widens
like a random walk's.

The weighting was chosen on the practice data by leave-one-out over the comparables themselves --
each launch in turn forecast from the others and scored with the contest's pinball loss: flat
pooling, empirical or normal, scored about 0.71-0.78, the hour-1-weighted normal about 0.39, and
the house persistence baseline about 0.93 on the same items. (The baseline's published practice
loss, about 0.387, is on the four practice launches, a different and tamer set of days.)

The program is deterministic: case files are read in name order, weights and sums accumulate in
file order, and nothing here uses randomness, the clock or the network, so two runs on the same
case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves this file's directory off sys.path; nothing else is imported from it, but
# keep the entry self-locating the way the kit's starter does.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The similarity kernel's bandwidth, in natural-log units of the first hour's size.
BANDWIDTH = 1.0
# The fallback's spread, in natural-log units per square-root hour.
FALLBACK_SIGMA = 0.15
# A comparable needs a first hour and at least this many later hours before its shape is trusted.
MIN_DAY_HOURS = 6
# An hour needs at least this many voting comparables before the weighted fit is used for it.
MIN_VOTES = 3
# Every written quantile is kept inside (0, MAX_Q].
MAX_Q = 1_000_000_000.0
MIN_Q = 0.01

NORMAL = NormalDist()


def parse_instant(text):
    """An ISO 8601 timestamp from the case file, as epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def round2(x):
    """x rounded to 2 decimals, half up; a whole number comes back as an int for compact JSON."""
    whole = math.floor(x * 100 + 0.5)
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def comparable_curves(comparables):
    """Each comparable's release day as ({hour: viewers}, ln(first hour)), skipping unusable ones.

    A comparable's release day is the 24 UTC hours starting at its `day_start`, and its hours are
    numbered as the case's are: hour 1 starts at `day_start`. Comparables with no positive hour-1
    count or with too few sampled hours are skipped.
    """
    curves = []
    for comp in comparables:
        try:
            day0 = parse_instant(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        day = {}
        for row in comp.get("hourly") or []:
            try:
                idx = int(round((parse_instant(row["from"]) - day0) / 3600.0))
            except (KeyError, TypeError, ValueError):
                continue
            viewers = row.get("viewers")
            if 0 <= idx <= 23 and isinstance(viewers, (int, float)) and viewers > 0:
                day[idx + 1] = float(viewers)
        first = day.get(1)
        if first is None or len(day) < MIN_DAY_HOURS + 1:
            continue
        curves.append((day, math.log(first)))
    return curves


def hour1_print(case):
    """The case's hour-1 print: its `known` row for hour 1, else the last hourly history row."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and isinstance(row.get("viewers"), (int, float)):
            return max(1.0, float(row["viewers"]))
    hourly = case.get("history_hourly") or []
    if hourly and isinstance(hourly[-1].get("viewers"), (int, float)):
        return max(1.0, float(hourly[-1]["viewers"]))
    return 1.0


def hour_quantiles(p1, ratios, weights, zs, h):
    """One hour's five quantiles: the weighted log-normal fit, or the persistence fallback."""
    if len(ratios) < MIN_VOTES:
        spread = FALLBACK_SIGMA * math.sqrt(h - 1)
        return [p1 * math.exp(z * spread) for z in zs]
    total = sum(weights)
    mean = sum(w * r for r, w in zip(ratios, weights)) / total
    var = sum(w * (r - mean) ** 2 for r, w in zip(ratios, weights)) / total
    sd = math.sqrt(max(var, 0.0))
    return [p1 * math.exp(mean + z * sd) for z in zs]


def forecast_case(case):
    """The forecast object for one case: {"hours": [{"hour": h, "q": [five quantiles]}, ...]}."""
    levels = case.get("levels") or LEVELS
    zs = [NORMAL.inv_cdf(tau) for tau in levels]
    p1 = hour1_print(case)
    log_p1 = math.log(p1)
    # Each comparable's release-day curve and its similarity weight to this case, from the
    # distance between the two first hours in log space.
    weighted = []
    for day, log_first in comparable_curves(case.get("comparables") or []):
        distance = (log_first - log_p1) / BANDWIDTH
        weighted.append((day, log_first, math.exp(-0.5 * distance * distance)))
    rows = []
    for spec in case.get("forecast") or []:
        h = spec["hour"]
        ratios = []
        weights = []
        for day, log_first, weight in weighted:
            viewers = day.get(h)
            if viewers is not None:
                ratios.append(math.log(viewers) - log_first)
                weights.append(weight)
        quantiles = hour_quantiles(p1, ratios, weights, zs, h)
        # Keep every quantile in range and the row non-decreasing, then round for compact output.
        cleaned = []
        previous = 0.0
        for q in quantiles:
            q = min(max(q, MIN_Q), MAX_Q)
            q = max(q, previous)
            previous = q
            cleaned.append(round2(q))
        rows.append({"hour": h, "q": cleaned})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
