#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 -- Monte Carlo forecast of the pump.fun top-5 by market cap.

Reads the house's recorded history (history.ndjson) up to the cut, fits a level +
fading-slope model per coin, projects each coin to every answer read, adds
noise that grows with lead time, simulates challenger coins born after the cut,
ranks all coins in every draw, and turns the counts into the probability output
the make.fun scorer expects. Python 3 standard library only, no network,
deterministic: the RNG is seeded from $MAKEFUN_TRIAL_SEED when set, else a
fixed constant; two runs produce byte-identical output.

Output shape (task.output):
  probabilities: [{mint, place, p}]  -- probability that mint finishes at place
  top5: [mint] x 5, in place order
  sizes: [{place, bin, p}]           -- log10-cap histogram of the coin at place
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------------

TRIALS = 6000          # Monte Carlo draws (each draw ranks all coins once)
SLOPE_WINDOW = 6       # marks used for the recent slope of a coin
SLOPE_CLIP = 0.5       # max |slope| in log10 units per hour
SLOPE_HALFLIFE = 2.0   # hours; the trend decays with this half-life
PULL_TO_MEDIAN = 0.15  # shrink projected level toward top-10 median level
PULL_PER_HOUR = 0.10   # extra shrink per hour of lead time (bounded)
BASE_SD_18H = 0.25     # noise sd (log10) at 18h lead
BASE_SD_24H = 0.35     # noise sd (log10) at 24h lead
YOUNG_SD_ADD = 0.08    # extra sd for coins with a short observed series
NOISY_SD_ADD = 0.12    # extra sd for coins whose recent residuals are wild
CAND_TOP = 30          # keep top-N of each mark as candidate coins
CAND_MARKS = 15        # over the last this many ok marks
CHALLENGERS = 3        # challenger coins born after the cut, per draw
CHALLENGER_PREFIX = "__challenger_"  # impossible as a real base58 mint
CHALLENGER_SPAWN_BIAS = -1.1  # challengers spawn below the current top 5
CHALLENGER_SPAWN_SD = 0.9     # spread of challenger spawn levels
CHALLENGER_SLOPE_CLIP = 0.7  # a newborn can moon, but not by 5 log10 units
CHALLENGER_NOISE = 0.08       # base jitter of a challenger at a read
SMOOTH_ALPHA = 1.0     # add-one (Laplace) smoothing on raw counts
PROB_FLOOR = 1e-4      # nothing may be exactly zero
MAX_NAMED_PER_PLACE = 100  # schema limit on rows per place
ROW_MIN_P = 0.005      # emit every named row at or above this p after
                      # tempering; a named coin with real mass beats dumping
                      # it into the unnamed share split over hundreds of coins
NAMED_ROW_FLOOR = 0.004  # a named row below this is a wasted row: the
                      # scorer's unnamed per-coin share is worth more than a
                      # 0.0002 row, so never emit one
TEMPER_ALPHA = 0.75    # p^alpha, renormalised to the place's previous mass;
                       # the standard correction for an over-confident simulation
MAX_P_FRAC = 0.45      # no named coin may take more than this fraction of its
                       # place's named mass; excess spread over the other named
                       # coins proportionally

SEED_ENV = "MAKEFUN_TRIAL_SEED"
FIXED_SEED = 20261112


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------

def parse_iso(s):
    """Parse an ISO-8601 UTC timestamp into epoch seconds."""
    if not s:
        return None
    if isinstance(s, datetime):
        if s.tzinfo is None:
            s = s.replace(tzinfo=timezone.utc)
        return s.timestamp()
    txt = str(s).strip()
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def load_inputs(input_dir):
    """Read task.json and history.ndjson; tolerate missing/broken pieces."""
    task = {}
    history = []
    task_path = os.path.join(input_dir, "task.json")
    if os.path.isfile(task_path):
        try:
            with open(task_path, "r", encoding="utf-8") as fh:
                task = json.load(fh)
        except (OSError, ValueError):
            task = {}
    if not isinstance(task, dict):
        task = {}
    hist_path = os.path.join(input_dir, "history.ndjson")
    if os.path.isfile(hist_path):
        try:
            with open(hist_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        history.append(obj)
        except OSError:
            history = []
    return task, history


def coin_cap_lamports(row):
    """Market cap in lamports as a float, or None when unusable."""
    if not isinstance(row, dict):
        return None
    v = row.get("market_cap_lamports")
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not (v > 0) or not math.isfinite(v):
        return None
    return v


def usable_marks(history):
    """Marks that are ok, carry rows, and are sorted by time; drop the rest."""
    out = []
    for m in history:
        if not isinstance(m, dict):
            continue
        if not m.get("ok"):
            continue
        rows = m.get("rows")
        if not isinstance(rows, list):
            continue
        at = parse_iso(m.get("at"))
        if at is None:
            continue
        parsed = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            mint = row.get("mint")
            cap = coin_cap_lamports(row)
            if not mint or cap is None:
                continue
            created = parse_iso(row.get("created_at"))
            parsed.append(
                {
                    "mint": str(mint),
                    "cap": cap,
                    "created": created,
                    "venue": row.get("venue"),
                }
            )
        parsed.sort(key=lambda r: -r["cap"])
        out.append({"at": at, "rows": parsed})
    out.sort(key=lambda m: m["at"])
    return out


# ----------------------------------------------------------------------------
# Candidate set and per-coin series
# ----------------------------------------------------------------------------

def build_candidates(marks, top_n=CAND_TOP, last_marks=CAND_MARKS):
    """Union of the top-N mints over the last `last_marks` usable marks.

    Returns a sorted list so downstream dict order never depends on the
    process-level string hash seed (byte-identical output requirement).
    """
    cands = set()
    for m in marks[-last_marks:]:
        for row in m["rows"][:top_n]:
            cands.add(row["mint"])
    return sorted(cands)


def build_series(marks, cands):
    """Per coin: (times in hours since the last mark, log10 caps, birth hour).

    The last usable mark is t=0; earlier marks are negative hours.
    """
    if not marks:
        return {}
    t_last = marks[-1]["at"]
    series = {mint: [] for mint in cands}
    for m in marks:
        t_h = (m["at"] - t_last) / 3600.0
        for row in m["rows"]:
            mint = row["mint"]
            if mint in series:
                series[mint].append((t_h, math.log10(row["cap"]) - 9.0))
    out = {}
    for mint, pts in series.items():
        pts.sort()
        out[mint] = pts
    return out


# ----------------------------------------------------------------------------
# Model fit: level, slope, noise
# ----------------------------------------------------------------------------

def robust_median(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def fit_coin(pts, slope_marks=SLOPE_WINDOW, slope_clip=SLOPE_CLIP):
    """Level at the last observation, recent slope per hour, residual spread.

    Returns (level, slope, resid_sd, age_hours, last_t) with level/last_t
    relative to the final mark (t=0). Coins observed once get slope 0 and a
    young age (which adds noise downstream).
    """
    if not pts:
        return None
    last_t, level = pts[-1]
    n = len(pts)
    age = last_t - pts[0][0]
    if n < 2:
        return (level, 0.0, None, age, last_t)
    win = pts[-min(n, slope_marks):]
    if len(win) < 2:
        return (level, 0.0, None, age, last_t)
    t0, c0 = win[0]
    t1, c1 = win[-1]
    span = t1 - t0
    if span <= 1e-9:
        slope = 0.0
    else:
        slope = (c1 - c0) / span
    slope = max(-slope_clip, min(slope_clip, slope))
    # residual spread around the local trend line
    resid = []
    for t, c in win:
        pred = c0 + slope * (t - t0)
        resid.append(c - pred)
    resid_sd = robust_median([abs(x) for x in resid])
    return (level, slope, resid_sd, age, last_t)


def median_level_of_top(marks, top=10):
    if not marks:
        return None
    caps = []
    for row in marks[-1]["rows"][:top]:
        caps.append(math.log10(row["cap"]) - 9.0)
    if not caps:
        caps = [math.log10(r["cap"]) - 9.0 for r in marks[-1]["rows"]]
    return robust_median(caps)


def cap_distribution(items, cap_frac):
    """Redistribute so no single item exceeds cap_frac of the total mass.

    Water-fill: p_i' = min(cap, p_i * s) with the scale s chosen so the sum
    is preserved (F(s) = sum min(cap, p_i*s) is monotone in s, so bisect).
    Deterministic: fixed iteration count, pure float arithmetic.
    """
    if not items:
        return items
    total = sum(p for _, p in items)
    if total <= 0.0:
        return items
    cap = cap_frac * total
    positive = [p for _, p in items if p > 0.0]
    if not positive:
        return items
    n_pos = len(positive)
    if n_pos * cap < total:
        # capping is infeasible (cap too small for the item count): the
        # flattest mass-preserving choice is uniform over the positive items
        share = total / n_pos
        return [(mint, share if p > 0.0 else 0.0) for mint, p in items]
    # bracket s: F(0) = 0 <= total; F(s_hi) = (#positive)*cap >= total
    # whenever cap_frac * count(positive) >= 1, which holds for >= 3 items at
    # 0.45; guard by widening if the bracket somehow fails.
    s_hi = cap / min(positive)
    for _ in range(4):
        if sum(min(cap, p * s_hi) for p in positive) >= total:
            break
        s_hi *= 2.0
    s_lo = 0.0
    for _ in range(200):
        s_mid = 0.5 * (s_lo + s_hi)
        if sum(min(cap, p * s_mid) for p in positive) >= total:
            s_hi = s_mid
        else:
            s_lo = s_mid
    s = 0.5 * (s_lo + s_hi)
    return [(mint, min(cap, p * s)) for mint, p in items]


# ----------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, history = load_inputs(input_dir)
    marks = usable_marks(history)

    places = task.get("places") or 5
    try:
        places = int(places)
    except (TypeError, ValueError):
        places = 5
    if places < 1:
        places = 1

    edges = get_edges(task)
    n_bins = len(edges) + 1

    reads = [parse_iso(r) for r in (task.get("reads") or [])]
    reads = [r for r in reads if r is not None]
    if not reads:
        t_last = marks[-1]["at"] if marks else 0.0
        reads = [t_last + 3600.0 * (i + 1) for i in range(7)]
    read_times = sorted(set(reads))

    seed_txt = os.environ.get(SEED_ENV)
    if seed_txt:
        try:
            seed = int(seed_txt)
        except ValueError:
            seed = FIXED_SEED
    else:
        seed = FIXED_SEED
    rng = random.Random(seed)

    # ---- degenerate: nothing usable ---------------------------------------
    if not marks:
        emit_degenerate(output_dir, task, places, n_bins, rng)
        return

    t_last = marks[-1]["at"]
    horizons = [(t - t_last) / 3600.0 for t in read_times]
    horizons = [h if h > 0.0 else 0.0 for h in horizons]

    # ---- candidates and fits ----------------------------------------------
    cands = build_candidates(marks)
    series = build_series(marks, cands)
    fits = {}
    for mint, pts in series.items():
        if pts:
            f = fit_coin(pts)
            if f is not None:
                fits[mint] = f
    if not fits:
        emit_degenerate(output_dir, task, places, n_bins, rng)
        return

    med_level = median_level_of_top(marks)

    # ---- projection to every read -----------------------------------------
    projections = project_all(fits, horizons, med_level)

    # ---- Monte Carlo -------------------------------------------------------
    results = run_simulation(rng, projections, horizons, places, n_bins, edges,
                              med_level, top_ref(marks))

    # ---- probabilities and output -----------------------------------------
    mints_sorted = sorted(projections.keys())
    emit_output(output_dir, task, places, results, n_bins, edges, mints_sorted)


def get_edges(task):
    size = task.get("size")
    edges = []
    if isinstance(size, dict):
        raw = size.get("edges")
        if isinstance(raw, list):
            for e in raw:
                try:
                    e = float(e)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(e):
                    edges.append(e)
    return edges


def top_ref(marks):
    """Median log10-cap level of the last mark's top 5 (or fewer) coins."""
    if not marks:
        return None
    caps = [math.log10(r["cap"]) - 9.0 for r in marks[-1]["rows"][:5]]
    if not caps:
        caps = [math.log10(r["cap"]) - 9.0 for r in marks[-1]["rows"]]
    return robust_median(caps)


def sd_for_lead(lead, age, resid_sd):
    """Noise sd in log10 units for a coin projected `lead` hours ahead.

    Grows like sqrt(time) up to 0.25 at 18h, then linearly to 0.35 at 24h.
    Young coins (short observed history) and noisy coins get extra spread.
    """
    if lead <= 0.0:
        sd = 0.04
    elif lead < 18.0:
        sd = 0.04 + 0.21 * math.sqrt(lead / 18.0)
    else:
        sd = 0.25 + (BASE_SD_24H - BASE_SD_18H) * (lead - 18.0) / 6.0
    if age < 1.0:
        sd += YOUNG_SD_ADD
    if resid_sd is not None and resid_sd > 0.30:
        sd += NOISY_SD_ADD
    return max(0.05, sd)


def project_all(fits, horizons, med_level):
    """For every coin: mean and sd of log10-cap at each horizon.

    Mean = level + fading-slope trend, pulled toward the top-10 median level.
    Sd   = sd_for_lead(lead) + young/short-history extra + noisy-series extra.
    """
    out = {}
    for mint, (level, slope, resid_sd, age, last_t) in fits.items():
        means = []
        sds = []
        for h in horizons:
            lead = max(0.0, h - last_t)
            decay = math.exp(-LEAD_LN2 * lead / SLOPE_HALFLIFE)
            trend = slope * SLOPE_HALFLIFE * (1.0 - decay) / LEAD_LN2
            mu = level + trend
            if med_level is not None:
                pull = PULL_TO_MEDIAN + min(0.25, PULL_PER_HOUR * lead)
                pull = min(0.6, pull)
                mu = (1.0 - pull) * mu + pull * med_level
            sds.append(sd_for_lead(lead, age, resid_sd))
            means.append(mu)
        out[mint] = (means, sds)
    return out


def run_simulation(rng, projections, horizons, places, n_bins, edges,
                   med_level, ref_top):
    """Draw TRIALS alternate futures; record who takes each place and how big."""
    mints = sorted(projections.keys())
    n_reads = len(horizons)

    # counts[place][mint] -> draws where that coin took that place
    counts = [dict() for _ in range(places)]
    # size_counts[place][bin] -> draws (cap of the coin at that place, named or not)
    size_counts = [dict() for _ in range(places)]
    size_total = [0] * places

    chal_mu = ref_top if ref_top is not None else (med_level or 3.0)
    chal_ids = [CHALLENGER_PREFIX + str(j) for j in range(CHALLENGERS)]
    field = mints + chal_ids  # ranked together, challengers stay unnamed

    for _ in range(TRIALS):
        # perturb every coin at every read: mint -> list of log10 caps
        draw = {}
        for m in mints:
            mu_list, sd_list = projections[m]
            caps = []
            prev = None
            for i in range(n_reads):
                x = rng.gauss(mu_list[i], sd_list[i])
                if prev is not None:
                    x = CORR * prev + (1.0 - CORR) * x
                caps.append(x)
                prev = x
            draw[m] = caps
        # challengers: coins born after the cut. They spawn below the current
        # top, carry a random fading trend (most fizzle, a few moon), and are
        # perturbed by lead-time noise just like real coins.
        for cid in chal_ids:
            spawn = chal_mu + CHALLENGER_SPAWN_BIAS + rng.gauss(0.0, CHALLENGER_SPAWN_SD)
            slope = min(CHALLENGER_SLOPE_CLIP,
                        max(-0.5, rng.gauss(0.25, 0.35)))
            caps = []
            prev = None
            for i in range(n_reads):
                lead = horizons[i]
                decay = math.exp(-LEAD_LN2 * lead / SLOPE_HALFLIFE)
                trend = slope * SLOPE_HALFLIFE * (1.0 - decay) / LEAD_LN2
                x = rng.gauss(spawn + trend, CHALLENGER_NOISE + 0.10 * math.sqrt(lead))
                if prev is not None:
                    x = CORR * prev + (1.0 - CORR) * x
                caps.append(x)
                prev = x
            draw[cid] = caps

        # rank all coins at each read, record the top `places`
        for i in range(n_reads):
            order = sorted(((draw[m][i], m) for m in field), reverse=True)
            for k in range(places):
                size_total[k] += 1
                if k >= len(order):
                    continue
                cap, who = order[k]
                counts[k][who] = counts[k].get(who, 0) + 1
                b = bin_of(cap, edges)
                size_counts[k][b] = size_counts[k].get(b, 0) + 1

    return {"counts": counts, "size_counts": size_counts, "size_total": size_total}


CORR = 0.5             # AR(1) correlation of a coin's noise across reads
LEAD_LN2 = math.log(2.0)


def bin_of(x, edges):
    """Bin index of x in the edge list; 0 below edges[0], len(edges) above."""
    lo = 0
    hi = len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def emit_output(output_dir, task, places, results, n_bins, edges, mints_sorted):
    """Turn counts into probabilities, then write task.output."""
    counts = results["counts"]
    size_counts = results["size_counts"]
    size_total = results["size_total"]
    out_name = task.get("output")
    if not isinstance(out_name, str) or not out_name:
        out_name = "forecast.json"

    probabilities = []
    for k in range(places):
        total = sum(counts[k].values())
        denom = total + SMOOTH_ALPHA * (len(counts[k]) + 1)
        if denom <= 0:
            denom = 1.0
        ranked = sorted(counts[k].items(), key=lambda kv: (-kv[1], str(kv[0])))
        # raw p over the real coins we may name, up to the schema's row limit
        # (challengers are never named)
        raw = []
        for mint, cnt in ranked:
            if mint.startswith(CHALLENGER_PREFIX):
                continue  # challenger coins are never named
            raw.append((mint, (cnt + SMOOTH_ALPHA) / denom))
            if len(raw) >= MAX_NAMED_PER_PLACE:
                break
        # temper: raise every named p to TEMPER_ALPHA and renormalise the
        # place to its previous mass. A sum of logs rewards calibration, not
        # confidence, so an over-confident spike at place 1 loses more on the
        # four thin places than it gains on the one it nails.
        mass = sum(p for _, p in raw)
        pow_sum = sum(p ** TEMPER_ALPHA for _, p in raw)
        tempered = []
        if pow_sum > 0.0 and mass > 0.0:
            scale = mass / pow_sum
            tempered = [(mint, p ** TEMPER_ALPHA * scale) for mint, p in raw]
        else:
            tempered = raw
        if mass > 0.0:
            tempered = cap_distribution(tempered, MAX_P_FRAC)
        # name every coin whose tempered p clears ROW_MIN_P: a named coin
        # with a small real p beats dumping that mass into the unnamed share,
        # which the scorer divides over the whole rest of the chart. Rows
        # under the threshold are simply not emitted; their mass stays in the
        # unnamed share (no renormalisation back into the kept rows).
        for mint, p in tempered:
            if p < ROW_MIN_P:
                continue
            if p < NAMED_ROW_FLOOR:
                p = NAMED_ROW_FLOOR
            if p > 1.0 - PROB_FLOOR:
                p = 1.0 - PROB_FLOOR
            probabilities.append({"mint": mint, "place": k + 1, "p": round(p, 6)})

    # top5: greedily take the most likely coin for place 1, 2, ... no repeats;
    # if the field is too small to fill five places, pad with the strongest
    # remaining candidate coins (mints_sorted) so top5 always has 5 entries.
    top5 = []
    used = set()
    for k in range(places):
        best = None
        for mint, cnt in sorted(counts[k].items(), key=lambda kv: (-kv[1], str(kv[0]))):
            if mint.startswith(CHALLENGER_PREFIX) or mint in used:
                continue
            best = mint
            break
        if best is None:
            continue
        top5.append(best)
        used.add(best)
    for mint in mints_sorted:
        if len(top5) >= places:
            break
        if mint not in used:
            top5.append(mint)
            used.add(mint)
    while len(top5) < places:
        filler = "UNKNOWN-%d" % (len(top5) + 1)
        top5.append(filler)
        used.add(filler)

    sizes = []
    for k in range(places):
        tot = size_total[k]
        if tot <= 0:
            continue
        for b in sorted(size_counts[k].keys()):
            p = size_counts[k][b] / tot
            p6 = math.floor(p * 1e6) / 1e6
            if p6 > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p6})

    payload = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=False, separators=(",", ":"))
        fh.write("\n")
    print("wrote %s (%d probabilities, %d top5, %d size rows)"
          % (path, len(probabilities), len(top5), len(sizes)))


def emit_degenerate(output_dir, task, places, n_bins, rng):
    """No usable history: name nothing (unnamed mass is shared by the scorer),
    keep a flat size histogram so every bin carries mass, and emit distinct
    placeholder mints so top5 still has `places` distinct entries."""
    out_name = task.get("output")
    if not isinstance(out_name, str) or not out_name:
        out_name = "forecast.json"
    probabilities = []
    sizes = []
    if n_bins > 0:
        p6 = math.floor(1e6 / n_bins) / 1e6
        for k in range(places):
            for b in range(n_bins):
                if p6 > 0:
                    sizes.append({"place": k + 1, "bin": b, "p": p6})
    payload = {
        "probabilities": probabilities,
        "top5": ["UNKNOWN-%d" % (i + 1) for i in range(places)],
        "sizes": sizes,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, out_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=False, separators=(",", ":"))
        fh.write("\n")
    print("wrote %s (degenerate: no usable marks)" % path)


if __name__ == "__main__":
    sys.exit(main())
