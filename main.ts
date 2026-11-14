// polymarket-faa-attribution-02 entry (Node 22, standard library only).
//
// Adapted from the polymarket-nvidia-ceo-forecast-01 entry (local practice
// pinball 0.2981 vs the house baseline's 0.3925 on the same transitions),
// with ONE deliberate change: the cross-market sum drift is REMOVED.
//
// Why: the NVIDIA bot's drift assumed its 24 markets were mutually
// exclusive CEO chances whose Yes prices sum to ~1, and drifted each
// market's centre toward (1 - S) * x / S. That assumption is false for
// this contest's final. The final is 4 FAA attribution markets (Iran / a
// foreign actor / a hack / an AI sandbox escape) that are NOT mutually
// exclusive -- Iran IS a foreign actor, and a sandbox escape could be a
// hack -- so their Yes prices have no reason to sum to 1. The practice
// cases are the NVIDIA event re-cut (24 markets, sums 0.81-1.09), so the
// practice score cannot see this: it rewards the drift (0.2981 with it,
// 0.3078 without). But on the universe the final is actually graded on,
// the drift is catastrophic. Measured in analysis/eval.py:
//   - real 4-market subsets of the practice cases (same transitions,
//     ground truth from the next case): drift as-is scores 3.7030 on the
//     mix4 subset (checkpoint sums 0.32-0.46) vs 0.4824 with the drift
//     off; top4 1.5124 vs 0.9104. Even drift_k = 0.25 scores 0.9968 on
//     mix4, still 2x worse than off.
//   - synthetic independent random-walk 4-market cases (the right null
//     model for non-exclusive markets), checkpoint sums 0.3-1.8, 4 seeds:
//     drift as-is pooled 2.17-2.23, drift off 0.38-0.41, starter 0.44-0.46.
//     Narrowing the drift to fire only when S is in [0.9, 1.1] still
//     loses to drift-off on 3 of 4 seeds (pooled 0.404 vs 0.396) -- when
//     S happens to sit near 1 by chance, a full-strength correction is
//     still a bet on exclusivity that the final does not offer.
// The practice score pays nothing; the final is what pays. So the drift
// goes, and the 0.0097 pp of practice score it bought goes with it.
// What remains -- and what beats the starter on every 4-market check --
// is the per-market part of the NVIDIA design:
//
//   - Persistence centre: the last price, no drift.
//   - Per-market spread from recent realised volatility: sigma_m =
//     clamp(0.0002 + 0.45 * RMS120, 0.00015, 0.0025), where RMS120 is the
//     root mean square of the last <=120 one-minute diffs, each winsorised
//     at 0.03. Prices are mostly flat with rare large jumps, so a market
//     that just moved gets a much wider interval than a dead one; the
//     floor keeps quiet markets from collapsing to a point forecast.
//   - Fat-tailed quantile multipliers, not normal z: [-3.0, -0.9, 0, 0.9,
//     3.0] times sigma_m * sqrt(h). The outer levels are wider than a
//     normal (jump risk), the inner levels narrower than the baseline's
//     (most minutes barely move). Revised 2026-11-14 from [-3.5, -0.75, 0,
//     0.75, 3.5] and a 0.02 winsorisation cap -- see "Revision" in
//     analysis/results.md: the narrower tails, slightly wider inner band
//     and looser cap improve the practice score and every 4-market check.
//   - Short histories are handled gracefully: RMS uses however many diffs
//     exist (a market with 3 prices still gets a sigma -- the final's
//     early cases hold only ~16h of history for markets created
//     2026-11-13); a market with no diffs falls back to the 0.0002 floor.
//     No randomness, no network, deterministic.
//
// What an entry must do (from starter/main.ts):
//   - read every <id>.in in the input directory (the grader runs one case
//     at a time, with only that case's file there), canonical JSON
//     {"schema":"makefun/polymarket-case/v1","case","at","horizon_minutes",
//     "quantiles","event","markets":[{"id","name","slug","yes_token","start",
//     "prices"}]}, where `prices` holds one Yes price per minute from `start`
//     to `at` inclusive, so the last one is the price at the checkpoint;
//   - write <id>.out to the output directory: {"forecasts":[{"id","q"}]},
//     every market of the case exactly once (no other id, no other key),
//     `q` exactly horizon_minutes rows, row h-1 the forecast for minute
//     at+h, each row one finite number in [0,1] per quantile level,
//     non-decreasing; at most 1 MiB.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Fat-tailed quantile multipliers (see header): outer levels wider than
 *  a normal's, inner levels narrower. Revised 2026-11-14: outer 3.5 -> 3.0
 *  and inner 0.75 -> 0.9 (see "Revision" in analysis/results.md) -- the
 *  narrower tails and slightly wider inner band win on practice and on
 *  every 4-market check. */
const Z = [-3.0, -0.9, 0, 0.9, 3.0];
/** Recent-volatility window, in minutes, for the per-market sigma. */
const VOL_WINDOW = 120;
/** Winsorisation cap for one-minute diffs when estimating volatility.
 *  Revised 2026-11-14: 0.02 -> 0.03 (see "Revision" in
 *  analysis/results.md) -- letting the rare 0.02-0.03 jumps count in full
 *  sharpens the vol estimate for markets that actually move, with no cost
 *  on the 4-market checks. */
const DIFF_CAP = 0.03;
/** Per-market sigma = clamp(SIGMA_FLOOR_A + SIGMA_B * RMS120, SIGMA_MIN, SIGMA_MAX). */
const SIGMA_A = 0.0002;
const SIGMA_B = 0.45;
const SIGMA_MIN = 0.00015;
const SIGMA_MAX = 0.0025;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Root mean square of the last <=VOL_WINDOW one-minute diffs, winsorised. */
function rms120(prices: number[]): number {
  const n = prices.length;
  if (n < 2) return 0;
  const start = Math.max(0, n - 1 - VOL_WINDOW);
  let sumSq = 0;
  let count = 0;
  for (let i = start; i < n - 1; i++) {
    const d = Math.abs((prices[i + 1] as number) - (prices[i] as number));
    const w = d > DIFF_CAP ? DIFF_CAP : d;
    sumSq += w * w;
    count++;
  }
  return count === 0 ? 0 : Math.sqrt(sumSq / count);
}

/** One market's forecast: horizon rows of the five quantiles. The centre is
 *  the last price (persistence); there is deliberately no cross-market
 *  drift -- see the header. Each market is forecast on its own prices
 *  only, so the program is identical for a 24-market practice case and a
 *  4-market final case.
 *  A market with no prices at all cannot be centred on anything; rather
 *  than throwing and invalidating the whole case (which would forfeit the
 *  entry's prize), it is forecast flat at 0.5 -- maximally uninformative,
 *  but the other markets still score. The case format guarantees a price
 *  per minute from each market's grid start, so this should never fire;
 *  it exists because the final's markets are hours old and a universe
 *  market could in principle arrive with no usable history. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  if (m.prices.length === 0) {
    return Array.from({ length: horizon }, () => [0.5, 0.5, 0.5, 0.5, 0.5]);
  }
  const x = m.prices[m.prices.length - 1] as number;
  const sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, SIGMA_A + SIGMA_B * rms120(m.prices)));
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const spread = sigma * Math.sqrt(h);
    rows.push(Z.map((z) => price6(x + z * spread)));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
