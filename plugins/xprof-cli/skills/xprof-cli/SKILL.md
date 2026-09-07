---
name: xprof-cli
description: Analyze local XProf profiles from the terminal with the official xprof CLI, using the skill-provided single-file, location-independent XSpace parser only as a fallback when the CLI cannot satisfy the requested analysis. Use when inspecting arbitrary-basename raw .pb files containing XSpace (including .xplane.pb and .xspace.pb), identifying their protobuf schema, filtering or aggregating raw events, summarizing JAX or TPU profile KPIs, inspecting HLO, memory, or LLO data, diagnosing kernel regressions, comparing captures, or producing bounded AI-readable evidence without relying on the XProf web UI.
---

# XProf CLI Analysis

## Overview

Use the official XProf CLI first. Only when it cannot satisfy the required
analysis, fall back to the skill-provided pinned-schema XSpace parser. Add derived XProf
views only when the capture contains the required data.

The examples use `uv run xprof` when the caller's repository already pins
`xprof==2.23.1`. Outside such an environment, replace that command prefix with
the portable pinned form `uv run --no-project --with xprof==2.23.1 xprof`.

Read the concise
[CLI reference](references/cli-command-reference.md) before invoking XProf in
the terminal. For the `tensorflow.profiler.XSpace` schema and direct parsing
without XProf, read the
[protobuf reference](references/xplane-protobuf-parser.md).

## Workflow

### 1. Resolve and protect the input

Locate arbitrary-basename raw `*.pb` XSpace files (including `*.xplane.pb` and
`*.xspace.pb`) or a TensorBoard profile session. Record all host files before
analysis.

Treat the source capture as immutable. Semantic commands can create
`ALL_HOSTS.*.pb` and `*.hlo_proto.pb` sidecars in the session directory. Import
a raw file into a unique scratch run before running them:

```bash
uv run xprof upload_trace \
  /tmp/xprof-logdir \
  /absolute/path/to/host.xplane.pb \
  --run_name=analysis-unique-id
```

For multi-host captures, import every host file into the same unique run. If a
session already contains meaningful sidecars, copy the complete session to a
scratch directory instead of importing only its XPlane files.

### 2. Pin and report the CLI version

Use the caller repository's locked XProf 2.23.1 environment when it has one.
Otherwise use the portable pinned invocation above. Resolve and report the
actual package version unless the user requests a version refresh:

```bash
uv run --no-project --with xprof==2.23.1 python -c \
  "from importlib.metadata import version; print(version('xprof'))"
```

Do not use `xprof --version`; XProf 2.23.1 does not implement it. An unpinned
external XProf invocation can change between runs. For a caller-repository
invocation, do not add `--active` merely to silence a mismatched `VIRTUAL_ENV`
warning. Always state the resolved version in an analysis report. When using
the standalone parser, report its pinned schema revision and
fingerprint plus the declared XProf compatibility boundary; it does not import
JAX or jaxlib.

### 3. Establish raw XPlane evidence

Use a direct protobuf path as `SESSION_ID` for the two raw event commands. The
leading `.` satisfies the CLI's `LOGDIR` argument but does not transform the
file:

```bash
uv run xprof list_xplane_events \
  . /absolute/path/to/host.xplane.pb \
  --plane_regex='TPU' \
  --event_regex='kernel-or-module-regex' \
  --max_events=50 \
  --bypass_cache=True

uv run xprof aggregate_xplane_events \
  . /absolute/path/to/host.xplane.pb \
  --plane_regex='TPU' \
  --event_regex='kernel-or-module-regex' \
  --bypass_cache=True
```

Keep `--max_events` bounded and paginate with `--offset`. Narrow aggregation
with both regexes because it scans at most 500,000 events. Interpret
`offset_ps` and `duration_ps` as picoseconds; divide by `1e6` for microseconds.
Do not infer cross-plane ordering from result order: compare timestamps.

#### Demo: mark and time one Pallas kernel

Prefer giving the kernel of interest a stable, distinctive marker before
capturing the profile. For a Pallas kernel, name both the call and any
fine-grained region that needs separate timing:

```python
import jax
from jax.experimental import pallas as pl


def kernel(...):
  with jax.named_scope("perf_chunk_kda_bwd_compute"):
    ...


call = pl.pallas_call(
  kernel,
  out_shape=...,
  name="perf_chunk_kda_bwd",
)
```

`pallas_call(name=...)` gives the call a stable debug name, while
`jax.named_scope(...)` can produce a finer-grained TPU trace region when
`--xla_enable_custom_call_region_trace=true` is present in `LIBTPU_INIT_ARGS`
before JAX initializes the TPU backend. Their final XPlane event names and
lines remain lowering-dependent, so list the actual events before choosing the
final selector. Add `--xla_xprof_register_llo_debug_info=true` when LLO debug
metadata is also required. `TraceAnnotation` and `StepTraceAnnotation` are
useful host markers or step windows, but they do not rename TPU device events
and must not replace module/custom-call timing.

After capture, run this demo from the caller repository's pinned environment,
or apply the portable command-prefix substitution above. Set `XPROF_PB` to the
host PB and make `XPROF_EVENT_RE` match the marker plus any observed
module/custom-call names:

```bash
XPROF_PB=/absolute/path/to/host.xplane.pb
XPROF_EVENT_RE='perf_chunk_kda_bwd|chunk_kda_bwd'
XPROF_DEMO_DIR=/tmp/xprof-perf-chunk-kda-bwd
mkdir -p "$XPROF_DEMO_DIR"

uv run xprof list_xplane_events \
  . "$XPROF_PB" \
  --plane_regex='TPU' \
  --event_regex="$XPROF_EVENT_RE" \
  --max_events=100 \
  --bypass_cache=True \
  > "$XPROF_DEMO_DIR/events.json" \
  2> "$XPROF_DEMO_DIR/events.log"

uv run xprof aggregate_xplane_events \
  . "$XPROF_PB" \
  --plane_regex='TPU' \
  --event_regex="$XPROF_EVENT_RE" \
  --bypass_cache=True \
  > "$XPROF_DEMO_DIR/timing.json" \
  2> "$XPROF_DEMO_DIR/timing.log"

for XPROF_RESULT in \
  "$XPROF_DEMO_DIR/events.json" \
  "$XPROF_DEMO_DIR/timing.json"
do
  jq -e \
    'if type == "object" and has("error") then error(.error) else . end' \
    "$XPROF_RESULT" > /dev/null
done

jq '[.[] | {
  event,
  count,
  total_duration_us: (.total_duration_ps / 1000000),
  avg_duration_us: (.avg_duration_ps / 1000000),
  min_duration_us: (.min_duration_ps / 1000000),
  max_duration_us: (.max_duration_ps / 1000000)
}]' "$XPROF_DEMO_DIR/timing.json"
```

For steady-state performance, warm up outside `jax.profiler.trace(...)`, run
multiple measured iterations inside it, and keep `block_until_ready()` inside
the trace. A one-event capture is useful for discovery but is not a stable
benchmark.

### 4. Fall back to direct XSpace parsing

`scripts/xprof-cli.py` is the only production script. It can be copied to any directory and invoked by absolute path; it imports no neighboring package and reads no repository-relative source, assets, or configuration. Its analysis inputs are the explicit CLI arguments and raw `.pb` paths, and its only implicit persistent state is the cache root, fixed at `~/.cache/pallas-kernel/xprof-cli` without a CLI override or XDG, platform, or CWD branching. Relative input paths and shell redirection targets still resolve from the caller CWD; use absolute paths when location independence matters.

Location independence does not mean stdlib-only. The script carries PEP 723 metadata for Python 3.12 or newer plus `protobuf` and `grpcio-tools`. The recommended `uv run /any/location/xprof-cli.py ...` resolves these dependencies without a repository checkout or project `pyproject.toml`; direct `python3` execution is valid only when compatible dependencies are already installed.

Only after XProf CLI cannot satisfy the required analysis, use the
single production script. Its local parsing core uses the XPlane schema pinned to
XProf 2.23.1 and imports neither JAX/jaxlib nor XProf. On the first cache miss
that needs this schema, the script downloads the pinned official OpenXLA
`xplane.proto` into `<cache-root>/xprof-cli-cache.v3/schema`, verifies its
expected SHA-256, and uses local `grpcio-tools==1.71.0` to compile the binding
there. Later cold builds reuse those artifacts offline. A hot SQLite hit never
creates, resolves, downloads, compiles, or imports the schema. The script never imports, invokes, or reuses official XProf. XProf 2.23.1 is only the declared compatibility and differential-test boundary; provenance always reports `official_xprof_reused:false`. All four commands query the local SQLite artifacts:

```bash
XPROF_CLI=/any/location/xprof-cli.py

uv run "$XPROF_CLI" \
  inspect /absolute/path/to/host.xplane.pb

# The total plane count remains complete; this only limits displayed summaries.
uv run "$XPROF_CLI" \
  inspect /absolute/path/to/host.xplane.pb --planes=:100

# Select one plane and keep its total line count while displaying a line slice.
uv run "$XPROF_CLI" \
  inspect /absolute/path/to/host.xplane.pb \
  --plane-index=1 --lines=100:200

# Aggregate event names and duration metrics on one known line.
uv run "$XPROF_CLI" \
  stats /absolute/path/to/host.xplane.pb \
  --plane-index=1 \
  --line-index=8 \
  --group-by=event-name \
  --metrics=duration \
  --limit=100

uv run "$XPROF_CLI" \
  events /absolute/path/to/host.xplane.pb \
  --plane='*TPU*' --event='*kernel-or-module*' --match=glob \
  --limit=50
```

All four local commands write one versioned JSON envelope to stdout. Use shell redirection when a file is needed; the CLI has no alternate renderer or file-writing branch.

Use `inspect` before choosing filters, `events` to discover stable locators,
`stats` for complete-scan aggregation, and `context --event-id LOCATOR` for
temporal neighbors. Check `analysis_complete`, `truncated`, `scan_complete`,
and `warnings` before accepting output. The schema revision and fingerprint are
pinned by the tool and reported by every command; the proto and generated
binding are cache artifacts, not production inputs, adjacent resources, or vendored source.

Event locators are scoped to the sole PB passed to the command and use
`plane:<N>/line:<N>/event:<N>`. They do not embed `profile_sha256`; keep the
separately reported profile hash when provenance must be verified. A legacy
`sha256:.../plane:...` locator is invalid under `xprof-cli.v3`.

For this single-file script, `profile_sha256` is the stable data identity across
commands and queries for the same raw bytes. The `events`/`stats` cursor
fingerprints bind that hash plus result-set, ordering, or aggregation settings;
page-size `--limit` is not bound and may change between pages. `inspect` always
reports the complete plane total in `profile.counts.planes`; it displays all
plane summaries by default, while `--planes=START:STOP` limits only the displayed
zero-based, half-open `plane_index` range. `--plane-index N` selects one plane
for line summaries; all lines are displayed by default, and
`--lines=START:STOP` limits only the displayed `line_index` range while
`selected_plane.line_count` remains complete. Raw `XPlane.id` remains reported
metadata but is not an inspect selector because it may repeat.
Its exit statuses are `0` for normal completion and `2` for an expected
argument/input/validation/cache/query failure. Expected failures write one plain
`xprof-cli: CODE: message` line to stderr and no JSON. Unexpected exceptions are
not wrapped: Python prints their traceback and exits `1`. A caught
`BrokenPipeError` still returns `0`, meaning the downstream consumer closed
stdout rather than proving complete delivery. Do not apply this mapping to
official `uv run xprof`, whose exit-code caveat is described in step 8.

Successful metadata reports cache status as only `hit` or `miss`. If a profile
database is corrupt or its identity check fails, stop concurrent invocations
that could use the same artifact, then use `mv` (preferred) to move its exact
profile-hash directory out of the cache tree for inspection, or use `rm -r` on
that exact directory. The directory is
`~/.cache/pallas-kernel/xprof-cli/xprof-cli-cache.v3/<first-two-hash-chars>/<profile-sha256>/`.
Rerun the command with the original `.pb`; the resulting miss rebuilds the
artifact without modifying the source profile.

On successful local JSON analysis envelopes, treat `analysis_complete`,
`scan_complete`, `truncated`, `profile_sha256_complete`, and
`raw_size_bytes_complete` as independent. Every analysis command accepts exactly
one raw XSpace `.pb`; a second positional input and source aliases/selectors are
unsupported. Failed invocations have no analysis envelope.

### 5. Run a bounded session first pass

Use the scratch logdir and run name created above:

```bash
uv run xprof get_hosts /tmp/xprof-logdir analysis-unique-id
uv run xprof get_overview /tmp/xprof-logdir analysis-unique-id
uv run xprof get_kpi_metrics /tmp/xprof-logdir analysis-unique-id
uv run xprof get_memory_profile /tmp/xprof-logdir analysis-unique-id
uv run xprof get_smart_suggestions /tmp/xprof-logdir analysis-unique-id
```

In OSS XProf 2.23.1, `get_hosts` reports labels derived from session filenames,
not `XSpace.hostnames`; do not use it as host-identity or clock-alignment
evidence.

Use overview and KPI data for orientation, not as a replacement for raw kernel
events. Keep metric definitions aligned when comparing captures. In
particular, do not compare an XPlane module window with HLO self-time as if
they were the same quantity.

### 6. Drill into HLO only after module discovery

List modules, select one explicitly, bound the text, and then inspect a named
instruction's producers and users:

```bash
uv run xprof list_hlo_modules \
  /tmp/xprof-logdir analysis-unique-id

uv run xprof get_hlo_module_content \
  /tmp/xprof-logdir analysis-unique-id \
  --module_name='module-name-from-list' \
  --max_lines=200

uv run xprof get_hlo_neighborhood \
  /tmp/xprof-logdir analysis-unique-id '%instruction.name' \
  --module_name='module-name-from-list' \
  --radius=2
```

Use `get_hlo_op_profile` or `get_top_hlo_ops` only when the capture and wheel
provide `hlo_op_profile`. If either returns missing-tool, `NoneType`, or error
data, report that boundary and fall back to static HLO plus raw XPlane timing.
Do not invent ranked HLO costs.

Avoid `get_graph_viewer` and unbounded `get_hlo_text` by default; a single
module can produce megabytes of text. Prefer `get_hlo_module_content` with
`--max_lines`, or `get_hlo_text --op_name=...` for a neighborhood.

### 7. Add memory and specialized diagnostics conditionally

Treat these as capture-dependent layers:

- Use `get_memory_profile` for capacity and peak HBM summary.
- Use `get_peak_allocations` only when memory-viewer/HLO allocation data exists.
- Use the three `detect_*` commands as experimental diagnostics and validate
  their payloads before accepting a no-bottleneck result.
- Use `get_llo_analysis` and `get_llo_debug_string` only when the capture was
  collected with LLO debug information.
- Use `get_utilization_viewer` only when utilization-viewer data is present.

Absence of a derived view does not invalidate raw XPlane events.

### 8. Validate stdout, stderr, and cache state

Separate XProf logs from result data:

```bash
uv run xprof get_overview \
  /tmp/xprof-logdir analysis-unique-id \
  > /tmp/xprof-overview.json \
  2> /tmp/xprof-overview.log

jq -e \
  'if type == "object" and has("error") then .error | error(.) else . end' \
  /tmp/xprof-overview.json
```

For official `uv run xprof` commands, do not trust exit status alone. Some
XProf 2.23.1 commands return exit code 0 with a JSON error object or a
plain-text error. Classify each result as valid JSON, expected text, binary
output, or failure before interpreting it.

Use a unique run name for every capture. Many commands cache results for 24
hours, and the OSS cache key can collide when the same session ID is reused
under different logdirs. Pass `--bypass_cache=True` where exposed when
rechecking a changed profile. Use Fire's capitalized boolean spelling;
lowercase `false` was observed to parse as truthy in XProf 2.23.1. The flag
bypasses only the CLI's SQLite cache, not native derived sidecars, so commands
without that flag and changed captures still require unique session IDs and
fresh scratch runs.

## Comparison Rules

- Use the same parser/tool version, filters, planes, event names, and units.
- Compare at least two captures when judging a kernel optimization.
- Report count, total, mean, min, max, and standard deviation for repeated
  kernel events; do not report only the fastest instance.
- Separate measured facts from inferences about layout, fusion, communication,
  or compiler behavior.
- State missing capture features and command failures explicitly.

## AI-Facing Deliverable

Return a compact report containing:

1. input files, hosts, session IDs, and exact parser/tool versions;
2. overview/KPI facts with units;
3. raw event evidence with plane, line, name, count, and duration statistics;
4. selected HLO or memory evidence, if available;
5. unsupported or failed views and their exact error category;
6. conclusions and next capture or code action, clearly marked as inference.

Keep raw JSON/text artifacts in a scratch directory and link or name them when
the user needs reproducibility. Do not paste an entire graph, HLO module, or
unbounded event list into the response.
