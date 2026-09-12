# XPlane PB protocol and standalone parsing

[中文翻译](xplane-protobuf-parser.zh.md)

## When to use it

Prefer the `xprof` CLI. Attempt direct PB parsing only when the CLI cannot
satisfy the current analysis requirement.

## Protocol identity

Both `*.xplane.pb` and ordinary `*.xspace.pb` files are binary protobufs. The
top-level message is not `XPlane`; it is:

```text
tensorflow.profiler.XSpace
```

The authoritative schema is in OpenXLA at:

```text
third_party/tsl/tsl/profiler/protobuf/xplane.proto
```

It uses `proto3` and the protobuf package is `tensorflow.profiler`.

Core hierarchy:

```text
XSpace
├── repeated XPlane planes
│   ├── repeated XLine lines
│   │   └── repeated XEvent events
│   │       ├── metadata_id
│   │       ├── oneof { offset_ps, num_occurrences }
│   │       ├── duration_ps
│   │       └── repeated XStat stats
│   ├── event_metadata[id] -> XEventMetadata
│   ├── stat_metadata[id] -> XStatMetadata
│   └── repeated XStat stats
├── errors
├── warnings
└── hostnames
```

`XStat.value` is another oneof containing `double`, `uint64`, `int64`,
`string`, `bytes`, or a `ref_value` that references a metadata name.

Binary protobuf stores field numbers and wire types, not field names and full
declared types. Exact interpretation requires a compatible `xplane.proto`;
never infer the schema from the suffix or a hex dump alone.

The planned stable CLI accepts any basename ending in `.pb` and always decodes
it with the pinned XSpace binding. A protobuf that fails the versioned XSpace
semantic check is reported as `INVALID_XSPACE`; an optional candidate message
type is only a best-effort hint, never a proven type. The design and exact
checks are defined in
[`xprof-cli-event-query-plan.md`](xprof-cli-event-query-plan.md#416-xspace-语义校验-check).

## Names and time

Event and stat metadata IDs are scoped to their owning `XPlane`:

```text
event.metadata_id -> plane.event_metadata[id].name/display_name
stat.metadata_id  -> plane.stat_metadata[id].name
```

Raw timing is:

```text
start_ps = line.timestamp_ns * 1000 + event.offset_ps
end_ps   = start_ps + event.duration_ps
```

If `XEvent.data` selects `num_occurrences`, the default `offset_ps=0` is not a
real start time. Confirm each line's time base before comparing planes. File
traversal order is not global time order.

The stable event model classifies offset-bearing records as `span` or
`instant`, `num_occurrences` records as `aggregate`, and records without valid
timing as `untimed`. Performance counters are XStats, not an XEvent kind.

## Direct-parsing fallback: skill-provided single file

Script: [`../scripts/xprof-cli.py`](../scripts/xprof-cli.py)

This is the only production source file. It can be copied to any directory and invoked by absolute path; it imports no adjacent package and reads no repository-relative source, assets, or configuration. Each command accepts one raw `.pb` path plus explicit CLI arguments, while its only implicit persistent state is the cache root, fixed at `~/.cache/pallas-kernel/xprof-cli` without a CLI override or XDG, platform, or CWD branching. The PEP 723 metadata lets `uv run /any/location/xprof-cli.py ...` resolve Python 3.12 or newer, `protobuf`, and `grpcio-tools` without a repository `pyproject.toml`; direct `python3` requires those dependencies to be installed already.

Its local parsing core uses the XLA revision pinned by XProf 2.23.1. The single production source file contains neither `xplane.proto` nor a generated `xplane_pb2.py`. The
core imports neither JAX/jaxlib nor XProf. The script never imports, invokes, or reuses official XProf; all four commands query local SQLite artifacts. XProf 2.23.1 is only the compatibility and differential-test boundary, and provenance reports `official_xprof_reused:false`. Every
invocation computes the raw bytes' SHA-256 and reports the parser revision and
schema fingerprint.

### Verified schema bootstrap

The schema pin is the official raw OpenXLA file at commit
`c520e3fb3f00ce8330d5088c08ee1a6f6067339f`:

```text
https://raw.githubusercontent.com/openxla/xla/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto
SHA-256: 25a5097f4a62c208ea2795c9b3cc1dc2919253fefb4469f4cd2496796124493c
```

The first cache miss that needs to decode XSpace downloads the exact file to a temporary path, verifies the digest, and compiles
`xplane_pb2.py` with local `grpcio-tools==1.71.0`. It validates the generated
descriptor against the pinned schema fingerprint before atomically publishing
both files under:

```text
<cache-root>/xprof-cli-cache.v3/schema/
├── xplane.proto
└── xplane_pb2.py
```

Later cache misses reuse this schema artifact without network access. A mismatched or incomplete artifact is never used,
and a download, digest, compiler, or descriptor failure aborts the cold build
without publishing a profile database. There is no fallback to an unpinned URL
or a latest schema. A compatible SQLite hit is checked before this bootstrap,
so it neither creates the schema directory nor loads a protobuf binding.

Successful metadata reports cache status as only `hit` or `miss`. If a profile
database is corrupt or fails its identity check, stop concurrent invocations
that could use the same artifact, then use `mv` (preferred) to move its exact
profile-hash directory out of the cache tree for inspection, or use `rm -r` on:
`~/.cache/pallas-kernel/xprof-cli/xprof-cli-cache.v3/<first-two-hash-chars>/<profile-sha256>/`.
Rerunning with the original `.pb` rebuilds the artifact on a miss without
modifying the source profile.

The reported `profile_sha256` is the stable data identity across commands and
queries over the same raw bytes. The `events` and `stats` cursor fingerprints
bind this hash and the settings that define the result set, order, or
aggregation. Page-size `--limit` is deliberately not bound and may change
between pages.

An `event_id` is scoped to the sole PB argument and has the form
`plane:<N>/line:<N>/event:<N>`. It does not repeat `profile_sha256`; use the
separately reported hash for provenance. The `xprof-cli.v3` grammar rejects the
legacy hash-prefixed locator.

### Successful output and the single input

Every successful full analysis envelope sets `profile_sha256_complete:true` and
`raw_size_bytes_complete:true`. Treat those fields, `analysis_complete`,
`scan_complete`, and `truncated` as independent. Every command accepts exactly
one raw XSpace `.pb`; a second positional input and source aliases/selectors are
unsupported. A failed invocation emits no analysis envelope.

For this single-file script only, process exit statuses are:

- `0` for normal command completion. A caught `BrokenPipeError` also returns
  `0`, meaning the downstream consumer closed stdout; it does not prove the
  complete output was delivered.
- `2` for an expected argument, input, validation, cache, or query failure.
  Stderr contains one plain `xprof-cli: CODE: message` line and stdout is empty.
- An unexpected exception is not wrapped; Python prints its traceback and exits
  `1`.

Successful commands write one versioned JSON envelope to stdout; use shell
redirection to save it because the CLI has no alternate renderer or
file-writing option. Expected failures do not write JSON. Do not apply this
mapping to official `uv run xprof` commands, which may return `0` with an error
payload.

List structure first:

```bash
XPROF_CLI=/any/location/xprof-cli.py

uv run "$XPROF_CLI" \
  inspect /abs/HOST.xplane.pb

# Keep the complete total, but display only plane_index 100 through 199.
uv run "$XPROF_CLI" \
  inspect /abs/HOST.xplane.pb --planes=100:200

# Select one plane and display only line_index 100 through 199.
uv run "$XPROF_CLI" \
  inspect /abs/HOST.xplane.pb --plane-index=1 --lines=100:200
```

`profile.counts.planes` always reports the complete plane total. Plane and line
ranges limit only their displayed summary lists. A selected plane likewise
keeps its complete `line_count`. Inspect selects it by unique `plane_index`;
raw `XPlane.id` remains reported metadata but is not a selector because it may
repeat.

Then return bounded events:

```bash
uv run "$XPROF_CLI" \
  events /abs/HOST.xplane.pb \
  --plane='*TPU*' \
  --line='XLA Modules' \
  --event='*target*' \
  --match=glob --limit=50 \
  > /tmp/xplane-events.json
```

Aggregate selected events without an implicit scan limit:

```bash
uv run "$XPROF_CLI" \
  stats /abs/HOST.xplane.pb \
  --plane='*TPU*' --event='*target*' --match=glob \
  --group-by=plane,line,event \
  > /tmp/xplane-aggregate.json
```

`events` globally sorts normalized records; its cursor fingerprint binds
`profile_sha256`, selectors, and sort configuration. The `stats` fingerprint
also binds grouping, metrics, percentile mode, and scan limit. Both deliberately
exclude page-size `--limit`. `stats` scans
completely by default; only an explicit `--scan-limit` makes `scan_complete`
false. Typed XStats are normalized before filtering and statistics. Public
EventRecords bound per-record XStats/flow/diagnostics with explicit
total/returned/truncated fields. Bytes are summarized by size, SHA-256, and a
bounded prefix in public output; the cache remains private.

## When raw fields are required

The cache-generated binding preserves raw IDs, oneof presence,
`errors/warnings`, metadata bytes, and integer-picosecond timing. A developer
changing the supported revision must review and update the official URL,
source digest, parser revision, descriptor/schema fingerprints,
`grpcio-tools` version, cache schema version, and schema tests together. A
runtime cache artifact must never be copied back into production source.

## Boundaries

- Each command streams the complete PB once for SHA-256. Cache misses then read
  and decode the complete message, so large files need substantially more
  memory than their file size. A cold build needs network only when its verified
  schema artifact is absent; compatible SQLite hits avoid schema access and
  protobuf object expansion entirely.
- Do not parse `*.hlo_proto.pb` as XSpace; it uses a different HLO schema.
- `*.trace.json.gz` is converted Trace JSON, not protobuf.
- Analyze per-host PB files in separate invocations; the script does not perform
  cross-profile queries. If one XSpace lists multiple hostnames, `inspect`
  reports that raw metadata and cross-line vertical context is unavailable
  because plane-to-host clock assignment is unknown.
- All internal event timing is integer picoseconds.
- The schema evolves. If no matching revision is available, record the schema
  revision used and treat unknown or absent fields as an evidence boundary.

## Primary sources

- [Pinned OpenXLA `xplane.proto`](https://github.com/openxla/xla/blob/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto)
- [OpenXLA `ProfileData` visitor](https://github.com/openxla/xla/blob/main/xla/python/profiler/profile_data_lib.h)
- [JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html)
- [JAX v0.11.1 release](https://github.com/jax-ml/jax/releases/tag/jax-v0.11.1)
- [Protobuf binary encoding](https://protobuf.dev/programming-guides/encoding/)
- [Protobuf Python generated code](https://protobuf.dev/reference/python/python-generated/)
- [ProtoJSON format](https://protobuf.dev/programming-guides/json/)
