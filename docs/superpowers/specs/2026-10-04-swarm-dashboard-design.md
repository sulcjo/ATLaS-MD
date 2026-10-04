# Swarm dashboard: a full-screen in-job TUI for the swarm round

Status: spec (user, 2026-10-04). Display only: it never changes what the swarm samples or writes
for analysis.

## 1. Problem

Under `--tui-mode dashboard|interactive`, umbrella production draws a full-screen dashboard. Its
pieces are:
- `DistanceLogger.log`, which builds a context and renders it;
- `build_context` / `DashboardContext`;
- `render_screen`, which places a spine, a view and a footer.

The swarm round (epoch 0, here 174 unbiased members × 5 ns) has no `DistanceLogger`. Since PR #132
it reports one aggregated progress event as `gareus_production`. In dashboard mode,
`GuiProgressSink` therefore draws a single line:

```
[gareus_production] | [###...] | 443065/1428714 | 31.0% | wall 2h10m | eta 4h50m | ...
```

That single line leaves a lot out:
- which members are grafting, equilibrating, running, done or failed;
- per-GPU load;
- what the swarm has covered so far;
- the straggler that will actually set the round's end.

The ETA is a mean-member estimate, but the round ends when its slowest member does. The per-member
`print()` lines interleave with these frames in the SLURM log.

The window-centric dashboard can't be reused as is. `DashboardContext`, `spine_lines` and the
three views read windows, exchange, overlap and boosts, none of which exist in a swarm. Its layout
layer, however, is domain-free and is reused unchanged:
- `gareus/tui_screen.py`: `Panel`, `Row`, `allocate_rows`, `compose_rows`, `frame_tiers`,
  `trim_panel`;
- `gareus/dashboard/panels.panel`;
- `gareus/tui.py`: `write_tui_frame`, `make_progress_bar`, `_sparkline`, `_coverage_bar`,
  `format_duration`, `_ansi_truncate`.

## 2. Design

### 2.1 Live state: `SwarmRoundProgress` becomes the swarm's state holder

`gareus/swarm/round_progress.py` keeps today's API and emits, and gains per-member detail. All of
it is updated under the existing lock and kept small: O(members) plus bounded rings.

**Per member:**
- `phase`: `queued | graft_wait | grafting | equilibrating | production | done | failed`;
- `device` token;
- `t_start`, `t_phase` (wall time);
- production steps, a 60 s ring of (wall, steps) for that member's ns/day;
- final `status` and `ns_per_day` from `done.json`.

**New hooks**, each O(1) and cheap. They are called from `members.run_member` and the driver,
and every one is a no-op when `round_progress` is None:

| Hook | Call site |
|---|---|
| `member_started(id, device=...)` | `driver._run_one` (device token added) |
| `member_phase(id, "graft_wait")` | before `with _GRAFT_GATE` |
| `member_phase(id, "grafting")` | inside the gate |
| `member_phase(id, "equilibrating")` | at the first `step_fn` call |
| `member_phase(id, "production")` | at the first production `step_fn` |
| `add_frame(id, cv1, rg_nm, e2e_nm)` | in `measure_fn`, after the trace row values are computed |
| `member_finished(id, ok, status=..., ns_per_day=...)` | already called; gains the fields |

`add_frame` feeds the coverage counters below:
- a CV1 histogram, 48 bins over `plan_meta` `edges.cv1` [min, max] widened by 10 %;
- the same for Rg and e2e;
- a `bins`-shaped count of stratification cells (`plan_meta` `edges` / `bins`, default 4×3×3)
  that live frames have visited.

Counters, never frame lists, so memory is bounded.

**Events ring:** the last 50 `member_finished` / failure events (id, cell, seed, status, ns/day,
wall). These replace the interleaved per-member `print()` lines in dashboard mode. Outside
dashboard mode the prints stay, unchanged.

`snapshot() -> SwarmSnapshot` returns a frozen, plain-data copy (dataclass) taken under the lock.
Rendering never holds the lock.

### 2.2 Render: `gareus/dashboard/swarm_view.py`

`render_swarm_screen(snap: SwarmSnapshot, *, term_w, term_h, glyphs="unicode") -> str`. It is pure
and deterministic (wall clock taken from `snap.now`) and reuses the `tui_screen` layout.

**Spine** (fixed, not window-centric). Full tier 4 lines; compact tier 2 (when `frame_tiers`
gives the compact spine):
1. `SWARM round R · <run label> · <n> members · <timestep> fs · <bins> cells`
2. Overall bar (mean-member production %), plus `elapsed`, `ETA mean`, `ETA last` (see the
   projection panel) and aggregate ns/day.
3. Member counts by phase, with the graft-gate occupancy `graft 4/4 (+n waiting)`.
4. Alert line: failures so far, stalled members (no production steps for > 10 min while in
   production), coverage gaps.

**Panels**, in priority order (1 = kept first when the terminal is small):

| Panel | Priority | Content |
|---|---|---|
| members | 1 | One glyph per member in id order (`·` queued, `g` graft_wait / grafting, `e` equilibrating, `▁▂▃▅▇` production in fifths, `✓` ok, `✗` failed), wrapped to width, with a legend. 174 members fit in about 4 lines at 80 columns. ASCII fallback with `glyphs="ascii"`. |
| projection | 2 | `ETA mean` = remaining mean-member steps ÷ aggregate rate. `ETA last` = the slowest running member's remaining steps ÷ its own rate, plus queued members' full length ÷ the median member rate. The round ends at the later of the two. Also: median and slowest member ns/day, completions per 10 min as a sparkline, and when the analysis (swarm analyze) is expected to start. |
| devices | 3 | Per device token: running members, summed ns/day from the members' 60 s rings, and production % of its members. Flags a device whose ns/day is < 50 % of the median device. |
| coverage | 4 | CV1 / Rg / e2e live histograms as coverage bars (`_coverage_bar`) against the planned stratification edges; cells visited N / total. Answers "has the swarm left its seed strata yet". |
| events | 5 | Tail of the events ring, newest first (completions, failures with status). |

**Footer:** `[swarm] round R · HH:MM:SS`, plus `dropped: <keys>` when the allocator drops panels.
This mirrors `footer_line`. There is a single view, so there are no view tabs.

Every frame must fit `term_w - 2` × `term_h - 1`, the same contract as the umbrella dashboard.

### 2.3 Driving the frames

`SwarmRoundProgress._emit_locked` keeps emitting the progress event (JSONL and the external
monitor are unchanged). When `tui_mode in {"dashboard", "interactive"}`, it also schedules a
swarm frame:

- **Throttle.** `--dashboard-render-interval-sec` if > 0, else a swarm default of 2 s. A forced
  emit (`member_started` / `member_finished`, about 2 per member) does not bypass the frame
  throttle. Today it does, which made the old one-line frame churn 350 times per round.
- **Rendering off the members' path.** The member thread that triggers the frame takes the
  snapshot under the lock, then renders and writes outside it under a non-blocking
  `_render_lock`. If a render is in progress, it skips. A member thread never waits for the
  terminal.
- **The sink's one-line console bar is suppressed for the swarm in dashboard mode.** This
  reverses the PR #132 exception for `extra["swarm"]`: the swarm frame replaces the bar. In
  console and none modes, behaviour is unchanged.
- **Terminal size.** `shutil.get_terminal_size((160, 40))`, as the umbrella dashboard.
- **SLURM.** Frames go to the log as today (`tui_clear_enabled` ignores isatty by design). With
  the 2 s default, a 5 h round writes about 9,000 frames, each a few KB, i.e. about 20-40 MB. A
  log-mode throttle is out of scope. `--dashboard-render-interval-sec 30` bounds it if wanted.

### 2.4 Status sidecar for the external monitor (small, additive)

Every 15 s, plus on the round's end, the round writes `swarm/round_RRR/live_status.json` atomically
(tmp + replace). It holds:
- the phase counts;
- per-member phase and production % (lists of length n);
- per-device ns/day;
- `eta_mean_s` and `eta_last_s`;
- the cells-visited count.

`gareus_monitor.py` integration (showing members / straggler ETA in the external TUI) is out of
scope. The file is the hook for it.

## 3. Unchanged on purpose

- What members run, the order, and everything written for analysis (trace, frames, features,
  `done.json`).
- The progress JSONL event and its fields.
- The external monitor.
- The umbrella dashboard.
- Behaviour in console and none TUI modes.
- Resume: members already done count as `done` from `done.json`.

## 4. Tests

**State (`tests/test_swarm_dashboard_state.py`):**
- Phase transitions per hook.
- `add_frame` histogram and cell counts against plan edges.
- The snapshot is a detached copy.
- 60 threads × hooks concurrently: counts are consistent and no exceptions occur.
- Throttle: forced emits do not exceed one frame per interval.
- The render-lock skip path.
- No frame in console and none modes.
- Hooks are no-ops with `round_progress=None`.

**Render (`tests/test_swarm_dashboard_render.py`):**
- Fit property over `WIDTHS × HEIGHTS` from `test_dashboard_frame_fit`: lines ≤ term_h − 1 and
  width ≤ term_w − 2, for 0, 1, 174 and 500 members.
- No `nan` in any frame.
- `ETA last` ≥ `ETA mean` when a straggler exists.
- Device flagging.
- The members glyph grid count equals n.
- Golden snapshot `tests/golden/dashboard/swarm_140x45.txt` (ANSI stripped, clock normalised;
  regenerated with `GAREUS_UPDATE_GOLDEN=1`).

**Wiring:**
- `run_member` calls the phase hooks in order, against a fake simulation (the existing
  swarm-member test fixtures).
- In dashboard mode the sink prints no swarm bar line.
- `live_status.json` is written atomically and parses.

## 5. Cost and risk

**Cost** per frame on one member thread:
- the snapshot (O(members)) plus rendering, about 5-20 ms of Python;
- at most 0.5 Hz, so under 1 % of one core;
- the GIL contention seen in the 174-thread stall (PR #132's graft fix) is not re-entered:
  rendering is rare and outside the lock.

**`add_frame`** is O(1) per frame per member.

**Risk:** a render exception must never kill a member. It is caught, warned once, and frames stop
for the round; the progress events continue.
