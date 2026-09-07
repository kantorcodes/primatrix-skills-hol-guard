# XProf CLI 2.23.1: AI terminal PB analysis quick reference

## Goal

Use `uv run xprof` from a caller repository that pins `xprof==2.23.1` to turn
local `.xplane.pb`/`.xspace.pb` files into **bounded, verifiable terminal
evidence**. If no such environment exists, replace every `uv run xprof`
command prefix below with the portable pinned form
`uv run --no-project --with xprof==2.23.1 xprof`. Do not depend on the web UI,
modify the original capture, or put unbounded binary, timeline, HLO, or graph
output into the AI context.

## Identify the input first

A standard JAX profiler layout is:

```text
LOGDIR/plugins/profile/SESSION/HOST.xplane.pb
```

- For raw events, use only `list_xplane_events` or
  `aggregate_xplane_events`, with arguments `. /abs/HOST.xplane.pb`.
- For derived views such as overview, memory, HLO, or LLO, use
  `. /abs/SESSION/`; do not pass a PB file or a bare timestamp session name.

## AI workflow

### 1. Pin the version

```bash
uv run --no-project --with xprof==2.23.1 python -c \
  "from importlib.metadata import version; print(version('xprof'))"
```

### 2. Collect raw XPlane evidence first

```bash
uv run xprof list_xplane_events \
  . /abs/HOST.xplane.pb \
  --plane_regex='TPU|CPU' \
  --event_regex='target-regex' \
  --max_events=50 \
  --bypass_cache=True \
  > /tmp/xprof-events.json 2> /tmp/xprof-events.log

uv run xprof aggregate_xplane_events \
  . /abs/HOST.xplane.pb \
  --plane_regex='TPU|CPU' \
  --event_regex='target-regex' \
  --bypass_cache=True \
  > /tmp/xprof-aggregate.json 2> /tmp/xprof-aggregate.log
```

### 3. Prepare a unique session for derived views

Import only an isolated PB:

```bash
uv run xprof upload_trace \
  /tmp/xprof-logdir /abs/HOST.xplane.pb \
  --run_name=analysis-unique-id
```

Import multiple host files with the same unique run name. If a session already
contains sidecars, copy the complete session to a temporary directory instead
of copying only its PB files.

### 4. Run a bounded first pass

```bash
uv run xprof get_hosts . /abs/SESSION/ --bypass_cache=True
uv run xprof get_overview . /abs/SESSION/ --bypass_cache=True
uv run xprof get_kpi_metrics . /abs/SESSION/
uv run xprof get_memory_profile . /abs/SESSION/ --bypass_cache=True
```

### 5. List, select, then bound HLO

```bash
uv run xprof list_hlo_modules . /abs/SESSION/ --bypass_cache=True
uv run xprof get_hlo_module_content . /abs/SESSION/ \
  --module_name='exact-module' \
  --max_lines=200 \
  --bypass_cache=True
uv run xprof get_hlo_neighborhood . /abs/SESSION/ '%instruction' \
  --module_name='exact-module' \
  --radius=2 \
  --bypass_cache=True
```

Continue with utilization, HLO op profile, Memory Viewer, LLO, or diagnostic
commands only when the capture contains the corresponding data.

## Validate results

- stderr contains conversion logs; stdout contains the result. Always save
  them separately.
- For official `uv run xprof` commands, exit code 0 does not prove success:
  JSON may contain `error`, and text may begin with `Error`, `Unexpected`, or
  `Failed`.
- Validate JSON results first:

```bash
jq -e 'if type=="object" and has("error") then error(.error) else . end' \
  result.json
```

- `offset_ps` and `duration_ps` are picoseconds. `list_xplane_events` iterates
  by plane/line/event; it is not globally time-ordered.
- `aggregate_xplane_events` scans at most about 500,000 events. The limit
  counts scanned events, not matches, so narrow both the plane and event
  regular expressions.
- In OSS 2.23.1, `get_hosts` derives labels from session `*.xplane.pb`
  filenames; it does not read `XSpace.hostnames`. Treat it as a session-file
  inventory, not proof of protobuf host identity or clock alignment.
- Always pass `--output_path` to `get_xspace_proto`; otherwise it writes binary
  bytes to stdout.
- Spell Fire booleans as `True` or `False`; lowercase `false` may be truthy.
- Most commands cache for 24 hours. The cache key contains `session_id` but not
  `LOGDIR`; use a unique session and `--bypass_cache=True`.
- `bypass_cache` does not remove `ALL_HOSTS.*.pb` or `*.hlo_proto.pb`
  sidecars.

## Single-file local XSpace CLI

The following contract applies only to `scripts/xprof-cli.py`, the sole production script, not to the official `uv run xprof` commands documented below. The file can be copied anywhere: it imports no adjacent package and reads no repository-relative source or configuration. Explicit CLI arguments and raw `.pb` paths are its analysis inputs; its only implicit persistent state is the cache root, fixed at `~/.cache/pallas-kernel/xprof-cli` without a CLI override or XDG, platform, or CWD branching. Use absolute input paths and an absolute shell redirection target when the invocation must be independent of the caller CWD.

The script carries PEP 723 metadata for Python 3.12 or newer, `protobuf`, and `grpcio-tools`. `uv run` resolves those dependencies without a repository checkout or project `pyproject.toml`:

```bash
XPROF_CLI=/any/location/xprof-cli.py
uv run "$XPROF_CLI" --help
```

Direct `python3` execution requires compatible dependencies to be installed already. All four commands emit one versioned JSON envelope on stdout; redirect stdout to save it. There is no alternate renderer or file-writing option.

- Exit `0` means normal command completion. A caught `BrokenPipeError` also
  returns `0`, which means a downstream consumer closed stdout and does not
  prove the complete output was delivered.
- Exit `2` means an expected argument, input, validation, cache, or query failure;
  stderr contains one plain `xprof-cli: CODE: message` line and stdout is empty.
- An unexpected exception is not wrapped; Python prints its traceback and exits `1`.

Successful metadata reports only `hit` or `miss` as the cache status. If a
profile database is corrupt or fails its identity check, first stop concurrent
invocations that could use the same artifact. Use `mv` (preferred) to move the exact
`~/.cache/pallas-kernel/xprof-cli/xprof-cli-cache.v3/<first-two-hash-chars>/<profile-sha256>/`
directory out of the cache tree; otherwise run `rm -r` only on it. Rerun with the
original `.pb` to rebuild the artifact on a miss. Neither recovery action changes
the source profile.

The single-file CLI JSON `profile_sha256` is stable across commands and queries
for the same raw bytes. `events`/`stats` cursor fingerprints bind that hash and
the result-set, ordering, or aggregation settings. Page-size `--limit` is
deliberately not bound and may change between pages.

Successful full analysis envelopes set `profile_sha256_complete:true` and
`raw_size_bytes_complete:true`; these fields remain independent of
`analysis_complete`, `scan_complete`, and `truncated`. Failed invocations do not
emit an analysis envelope or perform identity-only reads of later inputs.

Every analysis command accepts exactly one raw XSpace `.pb`. A second positional
input and source aliases/selectors are unsupported.

Event locators are relative to that sole input and use
`plane:<N>/line:<N>/event:<N>`. The profile hash remains separate provenance and
cursor-binding data; it is not part of `event_id`. Legacy
`sha256:.../plane:...` locators are rejected by the `xprof-cli.v3` grammar.

`inspect` always reports the complete plane total in `profile.counts.planes`.
It displays all plane summaries by default; `--planes=START:STOP` limits only
the displayed zero-based, half-open `plane_index` range. `:STOP` and `START:`
omit one bound. The output records `plane_range`, `planes_returned_count`, and
`planes_truncated`.

Use `inspect --plane-index N` to select one plane and display its line
summaries. Raw `XPlane.id` remains reported metadata but is not an inspect
selector because it may repeat. The selected plane's `line_count` is always
complete. All line summaries are displayed by default, while
`--lines=START:STOP` limits only the displayed zero-based, half-open
`line_index` range and records `line_range`, `lines_returned_count`, and
`lines_truncated`.

## Command quick reference (26)

List registered commands with `uv run xprof --help`. Inspect one command with
`uv run xprof COMMAND -- --help`. Brackets below mark optional arguments; do
not type the brackets literally.

### Import and export

- `uv run xprof upload_trace LOGDIR FILE_PATH [--run_name=NAME]` — Import an
  isolated PB.
- `uv run xprof get_xspace_proto . SESSION_DIR/ --output_path=FILE` — Export a
  single-host XSpace; never omit `output_path`.

### Run information

- `uv run xprof get_hosts . SESSION_DIR/ [--bypass_cache=True]` — List
  filename-derived session host labels; this is not `XSpace.hostnames` or
  clock-alignment evidence.
- `uv run xprof get_overview . SESSION_DIR/ [--include_command=True]
  [--bypass_cache=True]` — Return performance and environment data.
- `uv run xprof get_kpi_metrics . SESSION_DIR/` — Return overview and memory
  KPIs.
- `uv run xprof get_profile_summary . SESSION_DIR/ [--bypass_cache=True]` —
  Return an HLO text summary; it requires `hlo_op_profile`, and its top 10 is
  not reliable.
- `uv run xprof get_device_information . SESSION_DIR/ [--bypass_cache=True]`
  — Return roofline hardware ceilings.
- `uv run xprof get_smart_suggestions . SESSION_DIR/ [--bypass_cache=True]` —
  Return rule-based suggestions; an empty list does not prove there is no
  bottleneck.

### Timeline and utilization

- `uv run xprof list_xplane_events . HOST.xplane.pb [--plane_regex=REGEX]
  [--event_regex=REGEX] [--start_time_ps=N] [--end_time_ps=N]
  [--max_events=N] [--offset=N] [--bypass_cache=True]` — Return a bounded raw
  event list.
- `uv run xprof aggregate_xplane_events . HOST.xplane.pb
  [--plane_regex=REGEX] [--event_regex=REGEX] [--bypass_cache=True]` — Return
  count and total/average/minimum/maximum/standard-deviation durations; scan at
  most about 500k events.
- `uv run xprof get_utilization_viewer . SESSION_DIR/ [--host=N] [--device=N]
  [--node=N] [--bypass_cache=True]` — Return HBM, ICI, Vector, Scalar, VMEM,
  XLU, and MXU utilization.

### HLO

- `uv run xprof list_hlo_modules . SESSION_DIR/ [--bypass_cache=True]` — List
  modules and possibly generate sidecars.
- `uv run xprof get_hlo_module_content . SESSION_DIR/ [--module_name=NAME]
  [--max_lines=N] [--print_metadata=True] [--bypass_cache=True]` — Return
  bounded module text; select the module explicitly.
- `uv run xprof get_hlo_neighborhood . SESSION_DIR/ INSTRUCTION [--radius=N]
  [--module_name=NAME] [--print_metadata=True] [--bypass_cache=True]` — Run an
  operand/user BFS; the parser depends on `%name = ...` syntax.
- `uv run xprof get_hlo_text . SESSION_DIR/ [--path=FILE] [--module_name=NAME]
  [--op_name=NAME]` — Return a fixed 2,000-line module or radius-2 operation
  neighborhood; supplying `path` still prints to stdout.
- `uv run xprof get_graph_viewer . --session_id=SESSION_DIR/
  [--module_name=NAME] [--output_type=TYPE] [--node_name=NAME]
  [--graph_width=N]` — Native passthrough with no size bound; redirect it to a
  file.
- `uv run xprof get_hlo_op_profile . SESSION_DIR/ [--top_n=N]
  [--bypass_cache=True]` — Return self-time, FLOPs, and bytes; requires
  `hlo_op_profile`.
- `uv run xprof get_top_hlo_ops . SESSION_DIR/ [--limit=N]
  [--category_filter=NAME] [--bypass_cache=True]` — Return independent rankings
  by time, FLOPs, and bytes.

### Memory

- `uv run xprof get_memory_profile . SESSION_DIR/ [--bypass_cache=True]` —
  Return capacity, peak, stack, heap, free, and fragmentation data.
- `uv run xprof get_peak_allocations . SESSION_DIR/ [--limit=N]
  [--min_size_mib=N] [--output_format=json|markdown]
  [--include_summary=True|False] [--aggregate_instructions=True|False]
  [--bypass_cache=True]` — Return per-module peak buffers; requires Memory
  Viewer data.

### LLO

- `uv run xprof get_llo_analysis . SESSION_DIR/ [--host=NAME]
  [--bypass_cache=True]` — Analyze LLO already present in the capture.
- `uv run xprof get_llo_debug_string . SESSION_DIR/ [--host=NAME]
  [--bypass_cache=True]` — Return an unbounded `debug_string`; save it before
  taking a bounded excerpt.

### Diagnostics

- `uv run xprof detect_layout_mismatch_copies . SESSION_DIR/ [--limit=N]` —
  **Unavailable in OSS 2.23.1** because `_fetch_debug_info` is missing.
- `uv run xprof detect_unfused_reshapes . SESSION_DIR/ [--limit=N]` — Can
  degrade a top-HLO failure into `false`.
- `uv run xprof detect_unnecessary_convert_reduce . SESSION_DIR/ [--limit=N]`
  — **Unavailable in OSS 2.23.1**, but may still return `false`.

### Web

- `uv run xprof server [--logdir=LOGDIR] [--port=N] [--grpc_port=N]
  [--hide_capture_profile_button=True] [--enable_tab_name_label=True]
  [--worker_service_address=ADDR] [--src_prefix=PATH]
  [--max_concurrent_worker_requests=N]` — Start HTTP and gRPC services. Their
  ports must differ; both bind wildcard addresses without authentication or
  TLS.

## Do not call

These names are not registered in 2.23.1:

```text
llo_load llo_query diff_sessions find_session get_events_db_session_root
get_kernel_stats get_avg_step_time detect_unfused_updates get_hlo_stats
```

After changing the installed version, re-audit the actual `cli_main()` commands
and signatures instead of applying documentation from `master`.
