# pumpfun-top5-forecast-01

Monte Carlo forecast of the 5 largest pump.fun coins by market cap at each of the 7 hourly answer reads. This entry targets the make.fun bounty `pumpfun-top5-forecast-01`.
It reads the house's recorded history up to the cut (`task.json` + `history.ndjson`), fits a level + fading-slope model per coin, projects every coin to each read with lead-time-growing uncertainty, simulates newborn challenger coins, and ranks the whole field over 6000 seeded draws.
Run it with `MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py` (Python 3 standard library only, no network, deterministic, well under 300s).
Output: `probabilities` per place, `top5` in place order, and `sizes` as an 82-bin log10-cap histogram per place, written to `$MAKEFUN_OUTPUT_DIR/forecast.json`.

Place-distribution shaping (order score): the raw Monte Carlo counts are
sharpened (p^1.35, renormalised to the place's named mass), capped so no single
coin holds more than 65% of a place's named mass (water-fill redistribution),
and every coin whose sharpened p clears 0.02 is named (up to 12 rows per place)
with a 0.004 floor per named row. Direction chosen from two measured server
scores (v1 -37.10 vs flattened v2 -38.20): flattening cost 1.06 nats, so the
simulation is under-confident at the top. The size histograms are
deliberately untouched relative to the first submitted entry.
