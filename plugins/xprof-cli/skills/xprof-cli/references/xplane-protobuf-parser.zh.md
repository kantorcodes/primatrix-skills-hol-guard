# XPlane PB 协议与独立解析

[English source](xplane-protobuf-parser.md)

> 本文是英文 protobuf 解析参考的中文同步翻译。命令、schema 名称和证据边界保持原样；若两者出现偏差，以英文原文为规范源。

## 何时使用

优先使用 `xprof` CLI。只有 CLI 无法满足当前分析需求时，才尝试直接解析 PB。

## 协议身份

`*.xplane.pb` 和普通 `*.xspace.pb` 都是二进制 protobuf。顶层 message 不是 `XPlane`，而是：

```text
tensorflow.profiler.XSpace
```

权威 schema 位于 OpenXLA：

```text
third_party/tsl/tsl/profiler/protobuf/xplane.proto
```

它使用 `proto3`，protobuf package 为 `tensorflow.profiler`。

核心层级如下：

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

`XStat.value` 是另一个 oneof，可包含 `double`、`uint64`、`int64`、`string`、`bytes`，或引用 metadata 名称的 `ref_value`。

二进制 protobuf 保存字段编号和 wire type，而不是字段名和完整声明类型。准确解释必须使用兼容的 `xplane.proto`；绝不能仅根据文件后缀或十六进制 dump 推断 schema。

规划中的稳定 CLI 接受任意 basename、以 `.pb` 结尾的文件，并始终使用固定的 XSpace binding 解码。不能通过版本化 XSpace 语义检查的 protobuf 返回 `INVALID_XSPACE`；可选 candidate message type 只能是 best-effort hint，不能表述为已证明类型。精确规则由 [`xprof-cli-event-query-plan.md`](xprof-cli-event-query-plan.md#416-xspace-语义校验-check) 定义。

## 名称与时间

Event 和 stat 的 metadata ID 作用域都限定在所属的 `XPlane`：

```text
event.metadata_id -> plane.event_metadata[id].name/display_name
stat.metadata_id  -> plane.stat_metadata[id].name
```

原始时间计算为：

```text
start_ps = line.timestamp_ns * 1000 + event.offset_ps
end_ps   = start_ps + event.duration_ps
```

如果 `XEvent.data` 选择的是 `num_occurrences`，默认的 `offset_ps=0` 并不代表真实开始时间。比较不同 plane 前，应确认每条 line 的时间基准。文件遍历顺序不是全局时间顺序。

稳定事件模型把具有 offset 的记录分类为 `span` 或 `instant`，把 `num_occurrences` 记录分类为 `aggregate`，把没有合法 timing 的记录分类为 `untimed`。performance counter 属于 XStat，不是 XEvent kind。

## 直接解析 fallback：随技能提供的单文件脚本

脚本：[`../scripts/xprof-cli.py`](../scripts/xprof-cli.py)

这是唯一的生产源码文件。它可复制到任意目录并通过绝对路径调用，不导入邻近 package，也不读取仓库相对路径下的源码、资源或配置。每个命令只接受一个原始 `.pb` 路径及显式 CLI 参数，唯一的隐式持久状态是固定在 `~/.cache/pallas-kernel/xprof-cli` 的 cache root；CLI 不提供覆盖参数，也不按 XDG、平台或 CWD 分支。PEP 723 metadata 使 `uv run /任意目录/xprof-cli.py ...` 无需仓库 `pyproject.toml` 即可解析 Python 3.12 及以上、`protobuf` 和 `grpcio-tools`；直接使用 `python3` 时必须已经安装兼容依赖。

本地解析核心使用 XProf 2.23.1 固定的 XLA revision；生产源码不包含 `xplane.proto` 或生成的 `xplane_pb2.py`，也不导入 JAX/jaxlib 或 XProf。脚本从不导入、调用或复用官方 XProf；四个命令都查询本地 SQLite artifact。XProf 2.23.1 仅是兼容性与差分测试边界，provenance 固定报告 `official_xprof_reused:false`。每次调用都会计算原始字节 SHA-256，并报告 parser revision 与 schema fingerprint。

### 经校验的 schema bootstrap

schema 固定为 OpenXLA commit `c520e3fb3f00ce8330d5088c08ee1a6f6067339f` 的官方 raw 文件：

```text
https://raw.githubusercontent.com/openxla/xla/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto
SHA-256: 25a5097f4a62c208ea2795c9b3cc1dc2919253fefb4469f4cd2496796124493c
```

第一次遇到必须解码 XSpace 的 cache miss 时，进程把该文件下载到临时路径并校验摘要，再用本机 `grpcio-tools==1.71.0` 编译 `xplane_pb2.py`。生成 descriptor 通过固定 schema fingerprint 校验后，两个文件才原子发布到：

```text
<cache-root>/xprof-cli-cache.v3/schema/
├── xplane.proto
└── xplane_pb2.py
```

后续 cache miss 可离线复用这一份 schema artifact。任何不匹配或不完整的 artifact 都不能使用；下载、摘要、编译器或 descriptor 校验失败时，中止 cold build 且不发布 profile 数据库，也不回退到未固定的 URL 或 latest schema。兼容的 SQLite hit 在 bootstrap 前完成检查，因此不会创建 schema 目录，也不会加载 protobuf binding。

成功 metadata 中的 cache 状态只有 `hit` 或 `miss`。如果 profile 数据库损坏或身份校验失败，先停止可能使用同一 artifact 的并发调用，再用 `mv`（优先）把对应的完整 profile hash 目录移出 cache tree 以便检查，或仅对该目录执行 `rm -r`：`~/.cache/pallas-kernel/xprof-cli/xprof-cli-cache.v3/<hash前两位>/<完整profile_sha256>/`。随后用原始 `.pb` 重新执行命令；本次 cache miss 会重建 artifact，不会修改源 profile。

输出中的 `profile_sha256` 是相同原始字节跨命令、跨查询稳定的数据身份。`events` 和 `stats` 的 cursor fingerprint 绑定该 hash 以及会定义结果集、顺序或聚合语义的配置；只改变页大小的 `--limit` 不受绑定，可在续页时改变。

`event_id` 以本次唯一 PB 参数为作用域，格式为 `plane:<N>/line:<N>/event:<N>`，不重复携带 `profile_sha256`；需要核验 provenance 时使用单独报告的 hash。`xprof-cli.v3` 会拒绝旧的 hash 前缀 locator。

### 成功输出与单一输入

每个成功的完整 analysis envelope 都返回 `profile_sha256_complete:true` 和 `raw_size_bytes_complete:true`。这些字段与 `analysis_complete`、`scan_complete` 和 `truncated` 相互独立。每个命令恰好接受一个原始 XSpace `.pb`；不支持第二个位置输入、输入别名或 source selector。失败调用不返回 analysis envelope。

以下进程退出状态只适用于这个单文件脚本：

- `0` 表示命令正常完成。捕获到 `BrokenPipeError` 时也返回 `0`，此时只表示下游消费者关闭了 stdout，不能证明完整输出已经送达。
- `2` 表示可预期的 argument、input、validation、cache 或 query failure；stderr 只包含一行 `xprof-cli: CODE: message` 纯文本，stdout 为空。
- 意外异常不包装，由 Python 打印 traceback 并以 `1` 退出。

成功命令统一向 stdout 写一个版本化 JSON envelope；需要保存时用 shell 重定向，因为 CLI 没有其他 renderer 或文件写入参数。预期失败不写 JSON。不要把该映射套用到官方 `uv run xprof`；官方命令可能在输出错误 payload 时仍返回 `0`。

先列出结构：

```bash
XPROF_CLI=/any/location/xprof-cli.py

uv run "$XPROF_CLI" \
  inspect /abs/HOST.xplane.pb

# 总数仍完整汇报，只展示 plane_index 100 到 199
uv run "$XPROF_CLI" \
  inspect /abs/HOST.xplane.pb --planes=100:200

# 选中一个 plane，只展示 line_index 100 到 199
uv run "$XPROF_CLI" \
  inspect /abs/HOST.xplane.pb --plane-index=1 --lines=100:200
```

`profile.counts.planes` 始终报告完整 plane 总数，plane/line 范围只限制各自的展示列表。选中 plane 的 `line_count` 同样始终完整。inspect 使用唯一的 `plane_index` 定位；原始 `XPlane.id` 仍作为 metadata 展示，但因可能重复而不作为 selector。

再返回有界 event：

```bash
uv run "$XPROF_CLI" \
  events /abs/HOST.xplane.pb \
  --plane='*TPU*' \
  --line='XLA Modules' \
  --event='*target*' \
  --match=glob --limit=50 \
  > /tmp/xplane-events.json
```

按解析后的 event 名聚合：

```bash
uv run "$XPROF_CLI" \
  stats /abs/HOST.xplane.pb \
  --plane='*TPU*' --event='*target*' --match=glob \
  --group-by=plane,line,event \
  > /tmp/xplane-aggregate.json
```

`events` 对规范化记录做全局稳定排序；它的 cursor fingerprint 绑定 `profile_sha256`、selector 和排序配置。`stats` fingerprint 还绑定 grouping、metrics、percentile mode 和 scan limit。两者都不绑定只改变页大小的 `--limit`。`stats` 默认完整扫描；只有显式 `--scan-limit` 才会令 `scan_complete` 为 false。typed XStats 在过滤和统计前完整规范化；公开 EventRecord 通过 total/returned/truncated 字段有界展示每条记录的 XStats、flow 和 diagnostics。公开输出中的 bytes 只显示大小、SHA-256 和有界前缀；内部 cache 保持私有。

## 何时需要原始字段

cache 中生成的 binding 保留原始 ID、oneof presence、`errors/warnings`、metadata bytes 和整数皮秒 timing。开发者若改变支持的 revision，必须一起 review 并更新官方 URL、source digest、parser revision、descriptor/schema fingerprint、`grpcio-tools` version、cache schema version 和 schema tests；不得把运行时 cache artifact 复制回生产源码。

## 边界

- 每个命令都会流式读取完整 PB 计算 SHA-256。cache miss 随后完整读取并解码 message，因此大文件需要明显高于文件大小的内存；只有缺少已校验 schema artifact 的 cold build 才需要网络，兼容的 SQLite hit 完全不访问 schema，也不展开 protobuf object。
- 不要把 `*.hlo_proto.pb` 当作 XSpace 解析；它使用不同的 HLO schema。
- `*.trace.json.gz` 是转换后的 Trace JSON，不是 protobuf。
- 每个 host 的 PB 分别调用分析；脚本不执行跨 profile 查询。若单个 XSpace 列出多个 hostname，`inspect` 报告这份原始 metadata；由于缺少 plane-to-host 时钟归属，跨 line 的 vertical context 不可用。
- 所有内部 event timing 都使用整数皮秒。
- Schema 会持续演进。如果找不到匹配 revision，应记录所用 schema revision，并把未知或缺失字段视为证据边界。

## 一手来源

- [固定版本的 OpenXLA `xplane.proto`](https://github.com/openxla/xla/blob/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto)
- [OpenXLA `ProfileData` visitor](https://github.com/openxla/xla/blob/main/xla/python/profiler/profile_data_lib.h)
- [JAX profiling guide](https://docs.jax.dev/en/latest/profiling.html)
- [JAX v0.11.1 release](https://github.com/jax-ml/jax/releases/tag/jax-v0.11.1)
- [Protobuf binary encoding](https://protobuf.dev/programming-guides/encoding/)
- [Protobuf Python generated code](https://protobuf.dev/reference/python/python-generated/)
- [ProtoJSON format](https://protobuf.dev/programming-guides/json/)
