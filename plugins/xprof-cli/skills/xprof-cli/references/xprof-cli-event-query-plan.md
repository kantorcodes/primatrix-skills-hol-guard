# xprof-cli 设计文档

## 1. 定位

xprof-cli 是为了弥补官方 XProf 无法满足要求而进行开发的工具，其定位上不是官方功能的替代实现；基于此，应尽量与官方 XProf 的语义、数据模型和演进方向保持一致。

开发遵循以下规则：

1. 官方 XProf API 已经完整覆盖的能力，本项目不开发对应功能；用户直接使用官方 XProf。
2. 官方 API 只覆盖部分语义时，本项目只开发缺失部分，不重写已经被官方覆盖的部分。
3. 官方 API 没有相应能力时，完全由本项目实现。

目前聚焦两个核心需求：

1. 对任意的XSpace 中任意 XPlane、XLine、XEvent 做筛选、分组和统计以及索引。
2. 基于索引，定位任意指定XEvent，并查询：
   - 同一时刻其他 XPlane/XLine 上与它重叠的事件及其信息；
   - 同一 XPlane/XLine 上位于它前后、包含它或与它重叠的事件及其信息；

该工具首先服务于脚本化、可复现、输出范围可显式控制的终端分析。

## 2. 官方能力边界

- 当前版本固定为本地 XProf `2.23.1`，JAX/JAXLIB `0.11.1`。
- 本地 `xprof --help` 实际公开 **26 个命令名**：`cli_main()` 注册 25 个工具，加上 `server`。
- 官方参考：[OpenXLA/XProf](https://github.com/openxla/xprof)、[XProf 2.23.1 CLI registry](https://github.com/openxla/xprof/blob/xprof-v2.23.1/plugin/xprof/cli/xprof_cli.py)、[XProf 2.23.1 raw XPlane CLI 实现](https://github.com/openxla/xprof/blob/xprof-v2.23.1/plugin/xprof/cli/internal/oss/xplane_tools.py)、[XProf 2.23.1 session data 实现](https://github.com/openxla/xprof/blob/xprof-v2.23.1/plugin/xprof/cli/internal/xprof_data.py)、[XProf 2.23.1 OSS client 实现](https://github.com/openxla/xprof/blob/xprof-v2.23.1/plugin/xprof/cli/internal/oss/xprof_client.py)、[Trace Viewer](https://openxla.org/xprof/trace_viewer)、[XProf 2.23.1 固定的 XPlane schema](https://github.com/openxla/xla/blob/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto)。

| 命令 | 领域 | 具体功能与输出 | 数据依赖与能力边界 |
|---|---|---|---|
| `aggregate_xplane_events` | Raw events | 按 plane/event regex 汇总 count、total/avg/min/max/stddev duration。 | 最多扫描约 500,000 个事件；不支持 line/source 任意分组、分位数、区间并集、并发度或可靠 self time。 |
| `detect_layout_mismatch_copies` | 专项检测 | 检测两个 compute stage 之间由 layout mismatch 引起的 copy。 | OSS 2.23.1 缺少 `_fetch_debug_info`，实际不可用时不得把失败解释为“未检测到问题”。 |
| `detect_unfused_reshapes` | 专项检测 | 检测未融合 reshape/transpose/copy 导致的 HBM materialization。 | top-HLO/neighbor 获取失败时可能退化成 `false`，结果必须验证完整性。 |
| `detect_unnecessary_convert_reduce` | 专项检测 | 检测 bf16 reduce 前不必要的 f32 promotion。 | OSS 2.23.1 可能不可用或错误返回 `false`，不能把失败当作确定结论。 |
| `get_device_information` | Device/Roofline | 返回设备峰值 FLOP/s、带宽、ridge point 等硬件上限。 | 依赖 Roofline/device metadata；理论上限不等于实测利用率。 |
| `get_graph_viewer` | HLO/Graph | 获取 HLO graph short text、graph 数据或指定 node 周边视图。 | 需要 HloProto/Graph Viewer 数据；native passthrough 输出可能无界。 |
| `get_hlo_module_content` | HLO | 按 module 返回有界 HLO text，可选择 metadata 与最大行数。 | 必须先消歧 module；发生截断时必须明确报告。 |
| `get_hlo_neighborhood` | HLO | 以指定 instruction 为中心，对 operands/users 做半径受限 BFS。 | 依赖可解析的 HLO 文本和准确的 module 选择。 |
| `get_hlo_op_profile` | HLO profile | 返回高成本 HLO 的 self-time、FLOPs、bytes 等 profile。 | 依赖 `hlo_op_profile`；缺失时不能从静态 HLO text 猜测运行时排名。 |
| `get_hlo_text` | HLO | 返回 module 文本或指定 op 的局部文本。 | module 输出仍可能很大；`path` 参数不能保证完全避开 stdout。 |
| `get_hosts` | Input/session | 列出 session 中参与 profile 的 host。 | OSS 2.23.1 实际从 session 内 `*.xplane.pb` 文件名推导 host label，不读取 `XSpace.hostnames`；它既不能可靠枚举任意 basename 原始 `.pb` 内的 hostname，也不能证明各 host 时钟已经对齐。 |
| `get_kpi_metrics` | KPI | 汇总 overview、memory 等关键 KPI 为 JSON。 | 属于派生汇总，不替代 raw event 证据；依赖底层 tool data。 |
| `get_llo_analysis` | LLO | 对选定 host 的 XSpace 执行 LLO 分析并返回结构化结果。 | 需要 capture 中包含 LLO debug information。 |
| `get_llo_debug_string` | LLO | 返回 LLO analysis 的完整 debug string。 | 依赖 LLO debug sideband；输出可能无界，需要落盘或截断。 |
| `get_memory_profile` | Memory | 返回 memory capacity、peak、stack/heap/free、fragmentation 与 allocation timeline 摘要。 | 依赖 memory-profile tool data；不同设备的 allocator 语义可能不同。 |
| `get_overview` | Overview | 返回 performance、run environment、step、device 等综合 overview JSON。 | 指标取决于 capture 类型、设备和完整 step 数据。 |
| `get_peak_allocations` | Memory | 返回 peak 时刻的 module/buffer allocations，支持阈值、limit 和 instruction aggregation。 | 依赖 Memory Viewer/HLO buffer assignment；不是 raw XEvent duration 统计。 |
| `get_profile_summary` | Summary | 返回高层 profile/HLO 性能摘要。 | 依赖 `hlo_op_profile`；2.23.1 的 top-10 摘要存在可靠性边界。 |
| `get_smart_suggestions` | Diagnosis | 返回基于规则的优化建议。 | 空建议不证明没有瓶颈；只能作为启发式结果。 |
| `get_top_hlo_ops` | HLO profile | 分别按 time、FLOPs、bytes 排序，并支持 category filter/limit。 | 多个榜单口径彼此独立；依赖 HLO op profile 数据。 |
| `get_utilization_viewer` | Utilization | 返回 HBM、ICI、Vector、Scalar、VMEM、XLU、MXU 等 utilization 指标。 | 需要硬件 performance-counter sideband；并非所有设备或 capture 都有数据。 |
| `get_xspace_proto` | Input/export | 获取序列化 XSpace，可返回 binary 或写入 `output_path`。 | 官方实现主要是 session 导出接口；未指定路径时可能把二进制写到 stdout。 |
| `list_hlo_modules` | HLO | 列出 profile 中所有 HLO module，必要时生成 `*.hlo_proto.pb` sidecar。 | 需要 HLO metadata；生成 sidecar 会改变 session 目录。 |
| `list_xplane_events` | Raw events | 按 plane/event regex、时间范围、offset、limit 返回 raw XEvent JSON。 | 缺少 line filter、稳定 event ID、完整 XStats、全局时间排序和上下/左右 context。 |
| `server` | Service | 启动 XProf HTTP/gRPC 服务和 Web UI backend。 | 输入是 logdir、持续占用端口，不是单个原始 `.pb` 的离线分析。 |
| `upload_trace` | Input/workflow | 将 trace 文件导入指定 logdir/run 并返回状态。 | 会创建 session 并改变本地状态，属于导入 workflow 而非分析。 |

这 26 个命令定义了本项目的“不重复开发”边界：官方已经完整提供的能力保持由官方实现，本项目不另写同类功能。当前只补 raw-event 通用能力的明确缺口：`list_xplane_events` 没有 line selector、稳定 locator、完整 XStats、全局时间排序或上下/左右 context；`aggregate_xplane_events` 没有完整扫描保证、任意分组、分位数、区间并集、并发度和可靠 self time。对于包含多个 `XSpace.hostnames` 的单个 XSpace，schema 没有 event/plane 到 hostname 的映射；本工具只报告原始 hostname metadata，不做 host 归属、时钟对齐或跨 host 时间比较。

[Trace Viewer 的详细能力、数据来源、交互语义和 CLI 映射位于独立参考文档](xprof-trace-viewer-capability-matrix.md)。主设计只保留其核心约束：直接查询 XSpace/cache，不以完整 Trace Viewer JSON 为中间格式。

## 3. 需求拆解

本章只拆解**本项目需要补充的能力**。每项需求先判断官方 XProf API 的覆盖程度，再定义本项目必须补齐的缺口；“需要开发”始终表示开发缺失部分，不表示重写官方已经完整实现的功能。

### 3.1 需求、官方接口与开发缺口矩阵

本表中的“官方覆盖”只决定产品边界和差分测试基线，不表示运行时路由。单文件脚本从不导入、调用或复用官方 XProf；`inspect/events/stats/context` 一旦被调用，就只基于本次 `.pb` 与本地 SQLite 执行。若官方命令已完整满足用户需求，用户应在脚本之外直接调用官方 XProf。

本表直接完成 gap analysis。若一项能力已经被官方完整覆盖，则不应进入本表成为开发需求；表中的“是”表示仍有本项目必须补齐的缺口。

| 需求 ID | 我们的需求 | 官方相关接口（XProf 2.23.1） | 官方覆盖 | 是否需要开发 | 处理方式与备注 |
|---|---|---|---|---|---|
| `1` | 接受单个任意 basename 的原始 `*.pb`；当前契约只允许 XSpace 内容，并保持源文件只读。 | `get_xspace_proto` | 部分 | 是，仅补缺口 | 官方接口不能完成文件名无关的单文件输入契约与语义校验；本地只补 schema 校验、结构化错误和只读保证。 |
| `2` | 所有分析都以本次传入的原始 `.pb` 为事实源，用户无需提供 session/logdir 或手动导入。 | `upload_trace` | 不适用 | 否：边界约束 | 本脚本不调用 `upload_trace`、不构造 XProf session；只读本次 `.pb`，且不能让 cache 替代原始输入身份。 |
| `3` | 只保留唯一输入的 profile identity 和原始 `XSpace.hostnames` metadata；不检测、推断或执行跨 host/profile 时钟对齐。 | `get_hosts` | 不适用 | 否：边界约束 | OSS 2.23.1 `get_hosts` 只返回由 session 文件名推导的 label；本工具不调用它，也不把 `XSpace.hostnames` 当作 event 归属或时钟对齐证据。所有时间关系只在本次唯一 profile 的安全范围内计算。 |
| `4` | 按原始字节 SHA-256 建立可删除、可校验、可重建的 content-addressed cache。 | — | 无 | 是 | 实现单 SQLite artifact、临时构建与原子发布、权限、手动清理恢复和 cold/hot 一致性。 |
| `5` | 无损规范化并枚举任意 XSpace/XPlane/XLine/XEvent，事件具有稳定 locator。 | `list_xplane_events` | 部分 | 是，仅补缺口 | 官方接口只作为能力边界与差分基线；本地脚本独立解析并补齐 metadata 作用域、原始索引、完整 typed XStats 和稳定 locator。 |
| `6` | 提供 plane/line/event/stat/time 及可选 HLO 等 selector、确定性排序和分页。 | `list_xplane_events` | 部分 | 是，仅补缺口 | 本地脚本独立执行全部 selector；官方 plane/event/time 过滤只作为共同语义的差分基线。 |
| `7` | 对选中事件执行完整扫描和任意维度分组。 | `aggregate_xplane_events` | 部分 | 是，仅补缺口 | 本地脚本始终通过 SQLite 统计；官方结果只用于共同分组语义的差分验证。 |
| `8` | 计算 duration 分布统计。 | `aggregate_xplane_events` | 部分 | 是，仅补缺口 | 本地脚本计算全部公开指标；官方 count/total/mean/min/max/stddev 只作为共同语义的差分基线。 |
| `9` | 计算区间集合统计。 | — | 无 | 是 | 输出 wall span、active interval union、coverage 和 overlap factor，固定半开区间与 zero-duration 语义。 |
| `10` | 计算 interarrival、idle gap 和 average/max concurrency。 | — | 无 | 是 | 端点相同的排序与并发计数语义必须确定且可复现。 |
| `11` | 仅在合法嵌套 line 上计算 self/exclusive time。 | — | 无 | 是 | crossing interval 返回 unavailable 及原因，不能做错误相减。 |
| `12` | 聚合完整 typed XStats。 | `list_xplane_events` | 部分 | 是，仅补缺口 | 本地脚本完整保留 oneof，并实现数值汇总、string/ref 计数和 bytes 有界摘要；官方字段只用于差分。 |
| `13` | 通过 locator 或 selector 唯一定位 context 目标事件。 | `list_xplane_events` | 部分 | 是，仅补缺口 | 本地 SQLite 完成候选发现、稳定 locator、唯一性判断和 `AMBIGUOUS_TARGET`；官方过滤仅用于差分。 |
| `14` | 查询指定事件在其他 plane/line 上同一时刻的“上下”事件。 | —（Trace Viewer 仅提供 UI 交互） | 无 | 是 | 基于 interval overlap/point 语义返回关系、overlap、ratio、start/end delta；只在唯一 profile 的安全时间范围内执行，多-host XSpace 不执行跨 line/plane vertical context。 |
| `15` | 查询指定事件在同一 logical line 上时间前后的“左右”事件。 | —（Trace Viewer 仅提供 UI 交互） | 无 | 是 | 返回 before/overlapping/after、最近 predecessor/successor、gap、数量/时间窗口及稳定顺序。 |
| `16` | 将时间关系与 flow/因果关系分开。 | —（Trace Viewer 有 flow UI，无通用 CLI API） | 无 | 是 | `temporal_overlap` 不能冒充因果；`flow_related` 单独输出证据来源。 |
| `17` | 所有功能保持原始 `.pb` 驱动的单次离线 CLI 分析。 | `server` | 不适用 | 否：边界约束 | 持续 HTTP/gRPC 服务要求 logdir 和端口，不属于本工具接口。 |
| `18` | 所有结果通过 stdout 提供稳定、版本化且人类可读的 JSON，并显式标注单位、来源、完整性和截断。 | `list_xplane_events`、`aggregate_xplane_events` 的 JSON/dict 输出 | 部分 | 是 | 实现版本化 JSON envelope/schema、错误分类、provenance 和 bounded output。 |

本表是第 3 章唯一的需求 ID 来源。overview、KPI、HLO、memory、LLO、utilization 和 detector 等由官方完整提供的领域能力只保留在第 2 章能力审计中：不转换成本项目需求、不开发本地实现，也不在第 5 章重新暴露接口。

### 3.2 需要开发的需求

第 3.1 节中，需要本项目开发的需求 ID 是：

```text
1, 4-16, 18
```

这些 ID 只授权实现官方缺失部分。需求 `2`、`3` 和 `17` 是边界约束：用户无需执行 `upload_trace`，不提供 `server`，也不提供跨 host/profile 时钟对齐能力；它们不对应需要开发的分析算法。

| 开发范围 | 需求 ID |
|---|---|
| raw-PB 单文件输入与 cache | `1`、`4` |
| 统一事件模型、筛选与分页 | `5`、`6` |
| 任意事件统计 | `7`–`12` |
| 指定事件上下/左右文 | `13`–`16` |
| 统一可读输出 | `18` |

### 3.3 自研功能一：任意事件统计

功能一完全由第 3.1 节的需求 ID 组成。若用户只需要 `aggregate_xplane_events` 已完整覆盖的语义，应在本脚本之外直接调用官方命令；一旦调用本地 `stats`，全部 selector、分组和指标都由本地 SQLite 实现，不发生运行时转发。

| 需求 ID | 在功能一中的作用 |
|---|---|
| `1` | 接收并校验任意 basename 的原始 XSpace `.pb`。 |
| `2` | 保证原始 `.pb` 是事实源，不隐式导入 session。 |
| `3` | 保留唯一 profile identity 和原始 hostname metadata；不做 event-to-host 归属、跨 host/profile 聚合或时钟对齐。 |
| `4` | 以 profile SHA-256 复用解析结果和统计索引。 |
| `5` | 提供无损 EventRecord 和稳定 locator。 |
| `6` | 提供 selector、确定性排序、limit/cursor。 |
| `7` | 完整扫描并按任意维度分组。 |
| `8` | 计算 duration 分布。 |
| `9` | 计算区间并集、coverage 和 overlap factor。 |
| `10` | 计算 interarrival、idle gap 和 concurrency。 |
| `11` | 在合法嵌套条件下计算 self/exclusive time。 |
| `12` | 聚合 typed XStats。 |
| `17` | 保持单次 raw-PB 离线调用边界。 |
| `18` | 输出稳定、有界且人类可读的结果。 |

因此功能一的需求集合为：`1`–`12`、`17`–`18`；不包含只服务于 context 的 `13`–`16`。

#### 3.3.1 分组维度

允许按任意组合分组：

- plane/line；
- event name/display name/metadata ID；
- HLO op/module/category；
- stat key/value 或 event kind。

默认分组键必须包含 plane、line 和 event，避免同一 profile 内不同位置的同名事件被静默合并。

#### 3.3.2 时间统计

基础指标：

- count、total、mean、min、max、stddev；
- p50、p90、p95、p99；
- wall span；
- active interval union；
- coverage = active union / selected window；
- overlap factor = total duration / active union；
- interarrival 和 idle-gap 分布；
- average/max concurrency。

分位数必须标记 exact/approximate。区间并集和并发度采用固定的半开区间与相同时间端点处理语义；具体索引和算法见第 4.6 节。duration、interval、gap、concurrency 和 self-time 只使用具有合法 `offset_ps` 的 `span`/`instant` timeline 事件；`aggregate`/`untimed` 记录不进入这些时间指标，并在每组完整性信息中报告排除数量和原因。

#### 3.3.3 self/exclusive time

只有在以下条件满足时才计算：

- 事件属于同一 profile/plane/logical line；
- 该 line 的事件可以解释为合法嵌套 span；
- 没有无法消解的 crossing intervals。

否则输出 `self_time: unavailable` 及原因，不能用所有重叠事件简单相减。

#### 3.3.4 XStats 统计

- 数值：count、min/max、mean、分位数；
- 字符串/ref：distinct count、top values；
- bytes：size 分布，默认不展开完整 payload；
- ref/string metadata 解析失败时保留原始 ID 和 warning。

performance counter 是 XStat 的一种语义用途，不是 XEvent 的 `kind`。`num_occurrences` 事件按 `aggregate` 记录处理，不能标记为 counter，也不能与普通 timeline 样本静默混合。

### 3.4 自研功能二：指定事件的上下与左右

功能二同样只引用第 3.1 节的需求 ID。官方 Trace Viewer 提供可视化交互，但没有可供 CLI 直接调用的通用事件邻域 API。

| 需求 ID | 在功能二中的作用 |
|---|---|
| `1` | 接收并校验任意 basename 的原始 XSpace `.pb`。 |
| `2` | 保证原始 `.pb` 是事实源，不隐式导入 session。 |
| `3` | 时间关系只使用本次唯一 profile 的坐标；不推断或执行跨 host/profile 时钟对齐。 |
| `4` | 复用 EventRecord、logical-line 和 interval 索引。 |
| `5` | 提供无损 EventRecord 和稳定 locator。 |
| `6` | 发现候选目标并提供稳定排序、分页。 |
| `12` | 返回目标及相关事件的 typed XStats oneof 语义；公开展示达到嵌套上限时明确报告总数和截断。 |
| `13` | 唯一定位目标；歧义时返回候选而非默选。 |
| `14` | 查询跨 plane/line 同一时刻的“上下”事件。 |
| `15` | 查询同 logical line 时间前后的“左右”事件。 |
| `16` | 分离 temporal overlap 与 flow/因果关系。 |
| `17` | 保持单次 raw-PB 离线调用边界。 |
| `18` | 输出有界、可追溯的 context 报告。 |

因此功能二的需求集合为：`1`–`6`、`12`–`18`；不依赖只服务统计计算的 `7`–`11`。

#### 3.4.1 目标事件定位

推荐流程：

1. 用 `events` 按名称、时间、line 等条件发现候选；
2. 用户或调用方选择稳定 `event_id`；
3. `context --event-id ...` 做无歧义查询。

若直接 selector 匹配多个目标，返回候选列表和 `AMBIGUOUS_TARGET`，不能默选第一条。
`context` 的目标必须是具有合法时间位置的 `span` 或 `instant`。locator 指向 `aggregate`、`untimed` 或时间校验失败的记录时，返回 `EVENT_NOT_TIMED`，同时保留目标 locator、kind 和不可查询原因。

#### 3.4.2 “上下”：跨 plane/line 的同时刻事件

对有持续时间的目标事件，默认使用半开区间重叠：

```text
a.start_ps < b.end_ps && b.start_ps < a.end_ps
```

对 zero-duration instant 单独定义点语义：点位于 span 的 `[start, end)` 内即相关；两个同时间点 instant 可标记 `exact`。`aggregate`/`untimed` 记录没有时间点，不参与该关系计算。

返回关系类型：

- `exact`；
- `contains`；
- `contained_by`；
- `overlaps_start`；
- `overlaps_end`；
- `instant_inside`。

每条关系同时返回 `overlap_ps`、`overlap_ratio_target`、`start_delta_ps`、`end_delta_ps`、profile/plane/line/event/stats 以及可选 HLO 信息。

所有时间关系只在本次唯一输入 profile 内计算。本工具不接受第二个 profile，不推断或应用跨 host 时钟偏移，也不存在“对齐证据充分时开启”的分支。若一个 XSpace 列出多个 hostname，由于无法把 event/plane 归属到具体 host，跨 line/plane 的 vertical context 不可用并返回 `CLOCK_ALIGNMENT_UNKNOWN`；同 logical line 的 horizontal context 仍可使用该 line 自身的时间坐标。

#### 3.4.3 “左右”：同 line 的时间邻域

同 logical line 事件按 `(start_ps, end_ps, line_index, event_ordinal)` 排序，而不是 protobuf 遍历顺序；`line_index` 用于稳定区分重复 line ID 的物理 segments。输出分区：

- `before`；
- `overlapping`；
- `after`。

同时给出：

- 最近的不重叠 predecessor/successor；
- 与目标的 idle gap；
- 可配置数量窗口 `--before N --after N`；
- 可配置时间窗口 `--window-before/--window-after`；
- 是否包含嵌套 parent/child/sibling。

`before/overlapping/after.events` 分别按上述时间键返回；最近 predecessor/successor 是独立字段，不能通过倒置 `before.events` 的公开顺序表达，也不能因 `--before 0`/`--after 0` 而省略。只有 logical line 上的 positive-duration spans 构成无重复区间、无 crossing 的合法嵌套结构时，才标注 parent/child/sibling；存在相同区间或 partial crossing 时，时间关系仍可返回，但 `line_role` 必须显式 unavailable 并给出原因。

#### 3.4.4 时间关系与因果关系分离

`temporal_overlap` 仅表示时间相交；`flow_related` 表示 XProf flow/annotation 推导的因果关联。二者必须独立输出，不能因同一时刻就声称存在因果关系。

## 4. 架构设计

###  4.0 架构要求

1. 交付物为 `xprof-cli.py` 应该是个单文件脚本， 不应该加载其他本地模块。
2. 交付物可复制到任意目录并通过绝对路径/相对路径调用
3. 脚本本身应该是个无状态的，接受指定的文件和参数，返回指定的结果，多次执行相同参数不应该具有随即性。
4. 脚本唯一隐式持久状态是在固定目录存放的cache。
5. 脚本的输出统一写json stdout， 需要文件时由调用方使用 shell 重定向。
6. 脚本携带 PEP 723 inline metadata，显式声明 Python 3.12 及以上、`protobuf` 和 `grpcio-tools`。主调用方式为 `uv run ../xprof-cli.py ...`，不隐式依赖和导入其他没有指出的东西。
7. 脚本接受任意 basename、以 `.pb` 结尾的文件
8. 脚本只接受tensorflow.profiler.XSpace 格式的protobuf 文件，其他格式直接报错。
9. 每个分析命令必须且只能接受一个原始 `.pb`；不支持多文件输入、输入别名或跨 profile 查询。
10. 脚本应该使用sqlite 作为cache， 避免重复计算
11. 脚本使用sha-256 hash作为 db的key
12. cache db内部不要求人类可读
13. cache db 只是可删除、可重建的加速层，原始文件输入始终是唯一信息源

### 4.1 `tensorflow.profiler.XSpace` 大致格式

```text
tensorflow.profiler.XSpace
├── repeated XPlane planes
│   ├── int64 id / string name
│   ├── repeated XLine lines
│   │   ├── id / display_id / name / display_name
│   │   ├── timestamp_ns / duration_ps
│   │   └── repeated XEvent events
│   │       ├── metadata_id
│   │       ├── oneof data { offset_ps | num_occurrences }
│   │       ├── duration_ps
│   │       └── repeated XStat stats
│   ├── map<int64, XEventMetadata> event_metadata
│   ├── map<int64, XStatMetadata> stat_metadata
│   └── repeated XStat stats
├── repeated string errors
├── repeated string warnings
└── repeated string hostnames
```
具体看
- [XSpace/XPlane/XLine/XEvent schema：`xplane.proto`](https://github.com/openxla/xla/blob/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto)

### 4.2 时间、顺序与标识规则

| 规则 | 精确定义 |
|---|---|
| 绝对开始时间 | `start_ps = line.timestamp_ns * 1000 + event.offset_ps`。内部始终使用整数皮秒。 |
| 绝对结束时间 | `end_ps = start_ps + event.duration_ps`。 |
| aggregated event | oneof 选择 `num_occurrences` 时不是普通 timeline span；不得以默认 offset 构造伪时间。 |
| 遍历顺序 | protobuf 中 plane/line/event 顺序不是全局时间顺序；查询前按明确键排序。 |
| line identity | `XLine.id` 可重复并表示同一逻辑 timeline；物理 locator 仍保留 `line_index`，逻辑分析另行合并同 ID segments。 |
| event identity | 原始 schema 没有独立 event ID；单 PB 调用内的稳定 locator 使用 `plane_index + line_index + event_ordinal`。 |
| metadata identity | event/stat metadata ID 只在所属 `XPlane` 内有效。 |
| host metadata | `XSpace.hostnames` 只作为原始 profile metadata 保存；它不构成 event-level identity，也不作为跨 host/profile 时钟对齐证据。 |
。

### 4.3 cache 处理逻辑

```text
唯一生产脚本 xprof-cli.py
  + 显式 CLI 参数
  + 原始 .pb
  + 固定 cache root ~/.cache/pallas-kernel/xprof-cli
  -> 流式计算 SHA-256
  -> 查找兼容的本地 content-addressed cache
       cache hit  -> 打开 SQLite artifact；不访问 schema 或网络
       cache miss -> 确保 cache schema 目录中存在已校验 binding
                       缺失 -> 下载固定 proto -> 校验 -> 本地编译 -> 原子发布
                  -> mmap raw PB + memoryview 二次复核 size/SHA
                  -> Python generated protobuf API materialize 唯一完整 XSpace object
                  -> 单次 traversal 直接 INSERT 临时 SQLite
                  -> 构建权威 event、可选 enrichment 与查询索引
                  -> 原子发布 SQLite
  -> selector、统计与上下文查询
  -> stdout 中的单个版本化 JSON envelope
```

- 单个 `xprof-cli.py` 内含完整实现；本链路不加载邻近 package、仓库文件或 CWD 配置。
- 权威 event 层只规范化原始 XSpace 中实际存在的 XPlane/XLine/XEvent，不修改原始 profile。
- 可选 enrichment 层必须记录 provenance，并允许 flow、HLO detail 或 source 信息缺失。
- cache hit 与 cache miss 必须产生语义一致、顺序稳定的查询结果。
- schema bootstrap 只属于 cold path；生产源码不携带 proto/binding，SQLite hot path 不创建 schema 目录、不加载 protobuf object，也不访问网络。
- cold miss 在 schema binding 准备完成后，以 `mmap` + `memoryview` 二次复核 raw PB 的 size/SHA，再通过 Python generated protobuf API materialize 本次构建唯一的完整 XSpace object。generated API 不支持 repeated-message streaming，因此这一个 profile-sized object 是不可避免的 cold-path 内存边界；随后单次 traversal 直接 INSERT 临时 SQLite，不得再构造 raw `bytearray`、profile-sized normalized planes/events/diagnostics graph、全量 `EventRecord` 或内存索引。
- 缓存物理后端固定为 SQLite；cache schema version 通过目录隔离不兼容表结构。SQLite 是规范化 EventRecord、XStats、flow、diagnostics 和查询索引的唯一持久化存储，不得再建立全量内存 cache。

当前 XProf master 正在发展 Events DB/Parquet 方向，其字段包含 device/thread/start/end/self time、TF/HLO op、flow、source 等。内部规范可以与其对齐，但本工具不能依赖尚未稳定、也未覆盖所有任意 raw plane 的 API：

- [Events DB event utilities（审计提交）](https://github.com/openxla/xprof/blob/4f294cfac6c227daadff4676cd8fe067f37ad55e/xprof/convert/events_db/event_utils.h)

### 4.4 cache 架构

profile 的权威身份是原始 protobuf 字节的 SHA-256：

    profile_sha256 = SHA256(exact_raw_pb_bytes)

固定物理布局为：

    <cache-root>/
      xprof-cli-cache.v3/
        schema/
          xplane.proto
          xplane_pb2.py
        <sha256-prefix>/
          <profile-sha256>/
            profile.sqlite3

cache schema version 把不兼容的 SQLite 布局和 parser schema/compiler 契约放到不同目录；不再引入 profile artifact generation、manifest、外部 checksum 或 quarantine 目录。一个 profile hash 在一个 cache schema 下只有一个 SQLite 数据库。

v3 schema pin 是 OpenXLA commit `c520e3fb3f00ce8330d5088c08ee1a6f6067339f` 的官方 raw URL `https://raw.githubusercontent.com/openxla/xla/c520e3fb3f00ce8330d5088c08ee1a6f6067339f/third_party/tsl/tsl/profiler/protobuf/xplane.proto`，预期 source SHA-256 为 `25a5097f4a62c208ea2795c9b3cc1dc2919253fefb4469f4cd2496796124493c`。binding 固定由本机 `grpcio-tools==1.71.0` 生成，并通过 descriptor fingerprint 校验。

下载和生成都写临时文件，所有校验成功后才原子发布。并发 cold invocation 允许重复工作，但只能发布完成校验的 artifact。失败不得留下可复用的部分 artifact，且不得自动改用 master/latest schema。schema artifact 属于 cache，可删除并在下次 cold build 重建；它不是 SQLite 中的 profile 数据，也不是生产源码。

每个命令的链路固定为：

    open raw profile
      -> 流式计算 SHA-256，并用同一 FD 检查读取期间未变化
      -> profile.sqlite3 存在？
           是 -> 记录 SQLite 引用
           否 -> schema artifact 存在且校验通过？
                    否 -> 下载并校验固定 proto
                       -> 本地编译并校验 binding -> 原子发布 schema
               -> 解析 XSpace 并写临时 SQLite
               -> 关闭并校验写入
               -> os.replace 原子发布
      -> 打开 immutable SQLite FD
      -> 在该查询连接上校验 profile hash 和 raw size
      -> 执行 SQL
      -> 返回有界结果

规则：

- cache root 固定为 `~/.cache/pallas-kernel/xprof-cli`；CLI 不提供覆盖参数，也不读取 XDG 环境变量或根据平台、CWD 分支。
- cache key 不含原始绝对路径；相同内容移动或改名后命中同一数据库，任何字节变化都会得到新的 hash 路径。
- 原始 `.pb` 不复制进 cache。唯一输入的 source path 只属于本次 invocation provenance，不写入 content-addressed SQLite。
- SQLite 是唯一持久化数据层：profile metadata、plane、line、event、typed XStat、flow、enrichment、diagnostics 和 SQL index 都在数据库中。
- 数据库兼容性只由 cache schema 目录、`PRAGMA application_id/user_version` 与 profile hash/raw size 校验；不另写查询不读取的 artifact registry、schema fingerprint 或 row-count metadata 表。
- ProfileRecord 只是最小引用：profile hash、source path、raw size、database path 和本次 `hit`/`miss` 状态；不得携带 planes、lines、diagnostics、events 或 profile 级全量索引。
- EventRecord 是查询结果结构，只在最终事件页或有界 context 结果需要返回时从 SQLite hydrate；它不是 cache 容器。
- cold build 与 schema bootstrap 都使用临时 artifact 加原子替换，保证 reader 只会打开完整 artifact；已打开的 immutable SQLite FD 在原子替换时保持有效。
- 查询打开后只做 SQLite application/schema version 与 profile hash/raw size 身份检查，然后直接执行命令 SQL。不存在先读 summary DTO、关闭、再重复开库的命中链路。
- 已有数据库缺表、损坏或身份不符时，命令以 `CACHE_READ_FAILED:` 文本前缀报告并保留原始错误；普通查询不静默吞掉程序错误或自动 quarantine。恢复前必须停止可能使用同一 artifact 的并发调用；用 `mv`（优先）将 `~/.cache/pallas-kernel/xprof-cli/xprof-cli-cache.v3/<hash前两位>/<完整profile_sha256>/` 移出 cache tree 以保留现场，或仅对该目录执行 `rm -r`，然后用原始 `.pb` 重试。新的 cache miss 会原子重建 artifact，且不会修改源 profile。
- cache 目录和数据库默认只允许当前用户访问。cache 可安全删除；任何清理都不得修改原始 .pb。
- 冷构建解析 protobuf 时允许有短生命周期的 parser 对象；构建完成后，后续命令不得保留任何与 profile event/line/diagnostic 数量线性增长的 Python cache。

数据库关系语义：

- 每个原始 event 对应一个 event row，同值但 locator 不同的事件不合并。
- event 通过关系键引用 plane、line 和 event metadata；metadata-owned XStat 只保存一次，event-local XStat 保留 owner/ordinal。
- 可能超出 SQLite signed 64-bit INTEGER 的 start_ps/end_ps 使用可排序、可逆的 i128 key。
- locator、logical line、name/metadata/HLO、duration 和时间查询使用 SQL index。
- 旧 schema 路径不读取、不迁移，也不主动删除。

### 4.5 内部规范化事件模型（EventRecord）

定义不可变 `EventRecord`，至少包含：

```text
profile_sha256, source_path
plane_index, plane_id, plane_name
line_index, line_id, line_name, line_timestamp_ns
event_ordinal, metadata_id, name, display_name
start_ps, end_ps: nullable; duration_ps; num_occurrences: nullable
kind: span | instant | aggregate | untimed
timing_valid, timing_unavailable_reason
stats: typed key/value records
hlo, flow, source_info: optional enriched records with provenance
```

`XSpace.hostnames` 属于 profile-level 原始 metadata，只由 `inspect` 报告；它不进入 EventRecord，也不参与 selector、分组、event identity 或 context 时间对齐。

`kind` 由原始 XEvent oneof 和 duration 决定，不能根据名称猜测：

| 原始字段 | `kind` | 时间与查询语义 |
|---|---|---|
| `offset_ps` present，`duration_ps > 0` | `span` | 计算 `start_ps/end_ps`；参与 timeline、统计和 context。 |
| `offset_ps` present，`duration_ps == 0` | `instant` | 计算单点 `start_ps == end_ps`；使用点语义参与查询。 |
| `num_occurrences` present | `aggregate` | `start_ps/end_ps` 为 null；保留 occurrences 和原始 duration，但不得把 duration 解释为单次或 timeline duration。 |
| XEvent data oneof unset，或 timing 字段未通过局部校验 | `untimed` | `start_ps/end_ps` 为 null 并给出 reason；保留记录供诊断，不参与时间查询。 |

`stats` 对 `aggregate` 单独报告 `aggregate_record_count` 和 `num_occurrences_sum`；原始 `duration_ps` 仅以 source-reported 值展示，除非有 producer-specific 证据，否则不与 timed event 的 duration 分布合并。`untimed` 只进入记录数和诊断。performance counter 通过 typed XStat 表达，不增加 `counter` event kind。

稳定事件地址采用结构化 locator，而不是事件名：

```text
plane:<plane_index>/line:<line_index>/event:<event_ordinal>
```

每条命令已经由唯一位置参数绑定一个原始 `.pb`，因此 locator 只表达该 XSpace 内部的物理位置，不重复携带 `profile_sha256`。`profile_sha256` 仍作为独立 provenance、cache key 和 cursor 绑定字段输出。locator 只能结合本次传入的 `.pb` 解释；将同一 locator 用于另一文件时，坐标会相对于另一 XSpace 解析。原始路径不参与 locator，因此同一个 `.pb` 改名或移动后事件身份保持不变。旧 `sha256:.../plane:...` 格式不兼容并以 `INVALID_EVENT_ID` 拒绝；本次 locator grammar 变更将公共 schema 升级为 `xprof-cli.v3`。若 normalization 变化会影响 event ordinal，后续仍必须升级 locator/schema version，而不能静默重定向旧地址。

### 4.6 查询索引与算法策略

查询层直接复用 SQLite 中的规范化数据和持久化索引：

- locator 由 event 关系键精确定位；
- logical line、name/metadata/HLO 和 i128 time key 使用 SQL index 收窄候选；
- `events` 在 SQL selector 之后排序、计数和分页，只 hydrate 当前页；
- `context` 在 SQL 中按 logical line、时间关系和 flow key 取有界候选；
- `stats` 把选中行流式写入自动删除的磁盘 SQLite 工作库，在其中完成分组、排序与精确分位数；不保留全量 Python 事件或分组集合。

这些规范化数据和索引按 `cache_schema_version + profile_sha256` 持久化；后续所有命令直接查询 SQLite，而不是为每种报告重复解析 `.pb` 或重建内存索引。规范存储后端固定为 SQLite，不再保留排序数组、全量 `by_*` map 或 Parquet 切换路径。只有 cold miss 建库阶段允许上节所述唯一完整 XSpace object；单次 traversal 完成并发布 SQLite 后立即释放，不能把这一例外扩展成 profile-sized normalized Python graph。

SQLite cache 的 hot/query 读取边界是：不加载 schema binding 或 protobuf object；`inspect` 直接查询 profile summary，并默认返回全部 plane summary，也可通过 `--planes START:STOP` 在 SQL 中限制展示范围；指定唯一 plane 后默认返回其全部 line summary，也可通过 `--lines START:STOP` 在 SQL 中限制展示范围；`events` 直接查询 profile SQLite，`stats` 使用自动删除的磁盘 SQLite 工作库承接统计状态，`context` 的候选关系状态留在 profile SQLite。除作为 `inspect` 最终输出的 plane/line summary 列表外，内存中只允许存在 CLI 参数、当前有界结果页、context 窗口和常数级/有界的流式状态；不得全量 hydrate `EventRecord`、不得构建 profile 级全量内存索引。

历史 JSON/SQLite 测量只作为重构前基线。当前实现必须在相同输入和环境下重新记录 DB 体积、cold build、hot inspect/query wall time 与 peak RSS；生产查询路径不存在 full-materialize adapter，验收以 SQL pushdown 和有界 Python 内存为准。

## 5. 接口设计

### 5.1 公开接口边界与统一调用形式

公开契约只有单个 `xprof-cli.py` 及其 `inspect`、`events`、`stats`、`context` 四个 CLI 命令，以及它们的参数、退出状态、成功 JSON schema 和纯文本错误格式；脚本内 helper/data structure 与 cache 物理格式均不承诺兼容性。四个命令分别负责输入检查、事件发现、功能一和功能二。

所有可调用接口使用同一个形式：

```text
<CLI> COMMAND PROFILE.pb [OPTIONS]
```

其中 `<CLI>` 表示从任意 CWD 调用任意位置的该文件：

```bash
XPROF_CLI=../any/location/xprof-cli.py
uv run "$XPROF_CLI"
```

`uv run` 按脚本的 PEP 723 metadata 解析 Python 3.12 及以上、`protobuf` 与 `grpcio-tools`，不读取仓库 `pyproject.toml`。如果兼容依赖已经安装，也可用 `python3 "$XPROF_CLI"`。脚本移动后不需要同步复制任何 package、proto、binding 或配置文件。

顶层 `--help` 和 `--version` 是无需 profile 的元调用；其中本地 `--version` 只报告单文件 CLI/schema boundary，不代表官方 `xprof`（XProf 2.23.1 未实现 `xprof --version`）。除此之外，每个分析命令都必须显式传入且只能传入一个任意 basename 的原始 `*.pb`，其顶层 message 必须是 `tensorflow.profiler.XSpace`。第二个 profile positional argument 必须作为参数错误拒绝；profile hash、cache ID、session ID 或 logdir 均不能替代原始输入，JSON/gzip 或其他 protobuf message 不能作为分析输入。cache lookup/build 对用户透明。本次为不保留旧接口的重构：`--mode=structure/events/aggregate`、旧 regex 参数和旧输出 schema 均不注册兼容入口。

通用参数：

| 参数 | 适用范围 | 语义 |
|---|---|---|
| `PROFILE.pb` | 所有可调用命令 | 恰好一个任意 basename 的 XSpace `.pb`；每次调用重新计算该文件的原始字节 SHA-256。 |

cache root 固定为 `~/.cache/pallas-kernel/xprof-cli`，不提供路径覆盖或强制重建参数。成功结果中的 cache 状态只有 `hit` 或 `miss`，因此也不设计单独的 `cache` 命令或 `--cache-status` 开关。

### 5.2 核心事件接口

| 接口 | 最小用法 | 对应功能与需求 ID | 主要参数 | 主要结果 |
|---|---|---|---|---|
| `inspect` | `<CLI> inspect <PB> [--planes START:STOP] [--plane-index N] [--lines START:STOP]` | 公共输入与缓存前置能力；需求 `1`–`5`、`17`–`18` | 通用参数、可选 plane 展示范围、可选单 plane line 明细与展示范围 | 输入的 hash/schema/结构及 event-kind 计数、完整 plane 总数、所选 plane summary、选中 plane 的完整 line 总数及所选 line summary、profile identity、原始 hostname metadata、cache `hit`/`miss` 状态与 cache schema version。 |
| `events` | `<CLI> events <PB> [SELECTORS]` | 两个核心功能的事件发现入口；需求 `1`–`6`、`12`、`17`–`18` | selectors、`--sort`、`--limit`、`--cursor` | 确定性排序的 EventRecord、稳定 locator、typed XStats oneof 与显式嵌套截断、matched/returned count 和下一页 cursor。 |
| `stats` | `<CLI> stats <PB> [SELECTORS] [--group-by FIELDS]` | **功能一：任意事件统计**；需求 `1`–`12`、`17`–`18` | selectors、`--group-by`、`--metrics`、`--percentile-mode`、`--scan-limit`、`--limit`、`--cursor` | duration/interval/gap/concurrency/self-time/XStats 聚合，以及每组完整性和 exact/approximate 状态。 |
| `context` | `<CLI> context <PB> (--event-id LOCATOR \| SELECTORS) [OPTIONS]` | **功能二：指定事件上下与左右**；需求 `1`–`6`、`12`–`18` | `--axis`、`--before`、`--after`、`--window-before`、`--window-after`、`--overlap-limit`、`--include-flow` | 唯一 target；唯一 profile 安全范围内跨 line/plane 的同时刻关系；同 logical line 的 before/overlapping/after；独立的 flow 关系。 |

推荐调用链是：先用 `inspect` 验证输入和原始 profile metadata，再用 `events` 找到稳定 locator；随后把同一个原始 `.pb` 传给 `stats` 或 `context`。任何命令都可以直接冷启动 cache，不要求用户预先执行 `inspect`。

### 5.3 Selectors、分组与 context 参数

`inspect --planes START:STOP` 使用零基、左闭右开的 `plane_index` 范围，只控制 `profile.planes` 展示内容。省略参数时展示全部 plane；`:STOP` 从第一个 plane 开始，`START:` 展示到末尾。`profile.counts.planes` 始终报告完整 profile 的 plane 总数，不受范围影响；`plane_range`、`planes_returned_count` 和 `planes_truncated` 明确报告本次展示范围及完整性。

`inspect --plane-index N` 使用唯一物理位置选择 plane，并在 `profile.selected_plane` 中展示该 plane 的 line summary；原始 `XPlane.id` 仍作为 metadata 展示，但因可能重复而不作为 inspect selector。不存在该 index 时返回 `PLANE_NOT_FOUND`。选中 plane 后默认展示其全部 line；`--lines START:STOP` 使用零基、左闭右开的 `line_index` 范围，只限制 `selected_plane.lines`，且必须与 `--plane-index` 同时使用。`selected_plane.line_count` 始终是该 plane 的完整 line 总数，`line_range`、`lines_returned_count` 和 `lines_truncated` 报告展示范围及完整性。plane summary 的 `--planes` 与 line 明细 selector 相互独立，选中 plane 即使不在 `--planes` 范围内也仍返回其 line 明细。

`events`、`stats` 共用 selector；`context` 在未提供 `--event-id` 时也用同一组 selector 发现唯一 target：

| 维度 | 参数 | 语义 |
|---|---|---|
| locator | `--event-id LOCATOR` | `events/stats` 可重复传入并按 OR 选择已知事件；`context` 最多接受一个作为 target。 |
| plane | `--plane NAME`、`--plane-id ID`、`--plane-index N` | 名称匹配与稳定物理索引分开。 |
| line | `--line NAME`、`--line-id ID`、`--line-index N` | `line-id` 可命中同一 logical line 的多个物理 segment。 |
| event | `--event NAME`、`--metadata-id ID` | `NAME` 同时检查 name/display name；metadata ID 只在所属 plane 内解释。 |
| stat | `--stat NAME`、`--stat-value VALUE` | 对 typed XStats 过滤；比较失败返回类型诊断，不做隐式字符串转换。 |
| time | `--start-ps T`、`--end-ps T`、`--time-relation overlap\|contained\|starts-in` | 使用绝对整数皮秒和半开区间；默认 `overlap`。 |
| duration/kind | `--min-duration-ps T`、`--max-duration-ps T`、`--kind span\|instant\|aggregate\|untimed` | `aggregate`/`untimed` 不伪装为 timeline span；time selector 只匹配具有合法 start/end 的记录。 |
| 可选 enrichment | `--hlo-op NAME` | 只匹配具有可验证 HLO 关联的事件；数据缺失时返回 readiness warning，不猜测关联。 |

字符串默认 `--match exact`，需要时显式指定 `--match glob` 或 `--match regex`。不同维度之间使用 AND；同一参数重复出现时使用 OR。selector 始终先于排序、分页和统计执行。

`events` 默认按 `(timing_bucket, start_ps_or_zero, end_ps_or_zero, plane_index, line_index, event_ordinal)` 排序，其中 `timing_bucket` 依次为 timed `span/instant`、`aggregate`、`untimed`；null 时间不得参与数值比较。`--sort time|duration|name` 只改变主排序键，kind bucket 和稳定 locator 字段仍提供确定性 tie-breaker；cursor fingerprint 绑定 `profile_sha256`、selector 和排序配置，其中任一项改变后旧 cursor 必须拒绝。`--limit` 不改变结果集或顺序，因此不进入 fingerprint，可在续页时改变。

`stats` 参数语义：

| 参数 | 语义 |
|---|---|
| `--group-by FIELDS` | 可选任意组合；默认 `plane,line,event`。完整字段为：plane 维度的 `plane`、`plane-id`、`plane-index`、`plane-name`；line 维度的 `line`、`line-id`、`line-index`、`line-name`、`line-display-name`；event 维度的 `event`、`event-name`、`event-display-name`、`metadata-id`；XStat 维度的 `stat`、`stat-name`、`stat-type`、`stat-value`；以及 `kind`、`hlo`、`hlo-op`、`hlo-module`、`hlo-category`。`plane`、`line`、`event`、`stat`、`hlo` 是复合分组字段；其中 `plane`、`line`、`event`、`stat` 的身份必须递归保留当前 profile 及其全部上级 scope，`hlo` 至少保留 profile scope，避免省略显式父字段时静默跨 plane/line 合并。其余粒度字段允许调用者显式选择 profile 内跨 scope 聚合，例如只按 `event-name` 或 `hlo-op` 合并同名对象。即使省略 kind，timed、aggregate 和 untimed 子结果也必须分栏，不能把 aggregate duration 混入 timed 分布。 |
| `--metrics duration,interval,gap,concurrency,self-time,xstats` | 可选任意组合；默认计算全部适用指标，不适用项返回原因。 |
| `--percentile-mode exact\|approximate` | 默认 `exact`；选择 approximate 时必须输出算法、误差/压缩参数。 |
| `--scan-limit N` | 默认不设上限并完整扫描；显式设置后按 `(plane_index, line_index, event_ordinal)` 顺序最多评估 N 个索引候选，并始终返回 `scan_complete:false` 与 `SCAN_LIMIT_APPLIED`。它限制的是候选扫描，不是过滤后的匹配数。 |
| `--limit N --cursor TOKEN` | 只限制分组结果页，不改变扫描集合或统计值；`--limit` 不进入 cursor fingerprint，可在续页时改变。`stats` fingerprint 绑定 `profile_sha256`、selector、grouping、metrics、percentile mode 和 scan limit。 |

`context` 参数语义：

| 参数 | 语义 |
|---|---|
| `--event-id LOCATOR` | 推荐的唯一目标选择方式；格式为 `plane:<N>/line:<N>/event:<N>`，坐标相对于本次传入的唯一 `.pb`，且目标必须是 timing-valid `span`/`instant`。 |
| 不提供 `--event-id` | 使用通用 selector 发现 target；0 个返回 `EVENT_NOT_FOUND`，多个返回 `AMBIGUOUS_TARGET` 和候选 locator。 |
| `--axis both\|vertical\|horizontal` | 默认 `both`；`vertical` 是跨 line/plane 的同时刻关系，`horizontal` 是同 logical line 的前后与重叠关系。 |
| `--before N --after N` | horizontal 两侧各返回多少条，默认各 5 条。 |
| `--window-before D --window-after D` | 可选时间窗，与数量限制同时生效；`D` 接受 `ps/ns/us/ms/s`。 |
| `--overlap-limit N` | vertical、horizontal overlapping 和可选 flow 关系各自的结果上限；截断时报告总匹配数、排序和 `truncated:true`。 |
| `--include-flow` | 额外查询 flow 证据；结果与 temporal overlap 分栏。 |

### 5.4 输出、错误与有界性契约

- `inspect`、`events`、`stats`、`context` 成功时只向 stdout 写一个版本化 JSON envelope；不存在其他 renderer。可预期失败只向 stderr 写一行 `xprof-cli: CODE: message` 纯文本，stdout 为空；不构造 error envelope。
- 需要文件时由 shell 重定向 stdout；CLI 本身不提供文件写入分支。重定向失败属于 shell/OS 边界。捕获到 `BrokenPipeError` 时返回 `0`，它只表示下游消费者主动关闭 stdout，不能证明完整结果已送达。
- 本地 `<CLI>` 正常完成返回 `0`；可预期的 argument/input/validation/cache/query failure 以 `2` 退出。意外或内部异常不捕获、不改写，由 Python 打印原始 traceback 并以 `1` 退出。该映射只适用于这个单文件 `xprof-cli.py`，不适用于可能在错误 payload 下仍返回 `0` 的官方 `uv run xprof`。
- 每个成功的完整 JSON analysis envelope 都包含 `schema_version`、command、唯一输入路径、`profile_sha256`、原始字节数、cache `hit`/`miss` 状态、cache schema version、parser/XProf 版本和 provenance，并固定返回 `profile_sha256_complete:true` 和 `raw_size_bytes_complete:true`。`analysis_complete`、identity complete flags、`scan_complete` 和 `truncated` 相互独立。
- 成功输出声明 warnings/errors、时间单位、排序规则和截断状态。`events/context` 返回 matched/returned count 与 `truncated`；可分页结果返回 cursor；`stats` 另返回 `scan_complete`、扫描量和 approximation 说明。
- `inspect` 的 `profile.counts.planes` 永远是完整总数；默认 `profile.planes` 展示全部 plane。显式 `--planes START:STOP` 只限制展示行，并通过 `plane_range`、`planes_returned_count`、`planes_truncated` 以及顶层 `truncated` 报告范围和完整性。指定唯一 plane 后，`selected_plane.line_count` 永远是该 plane 的完整 line 总数；默认展示全部 line，显式 `--lines START:STOP` 只限制 line 展示行并以同样方式报告完整性。
- EventRecord 的 XStats、flow 和 record-local diagnostics 以及 stats 分组内的 XStat 类型列表均有独立的 count/returned/truncated 字段；长 XStat string value 返回大小、hash 和有界前缀，其他被截断文本带显式 `*_truncated` 标记。截断只能影响成功结果展示，不能改变 selector、统计或内部规范化数据。
- 输入错误文本至少以 `FILE_NOT_FOUND:`、`UNSUPPORTED_INPUT_FORMAT:` 或 `INVALID_XSPACE:` 开头，并在需要时包含 `PROTO_DECODE_FAILED`、`EMPTY_XSPACE` 或 `SEMANTIC_CHECK_FAILED`；目标错误使用 `EVENT_NOT_FOUND:`、`AMBIGUOUS_TARGET:` 与 `EVENT_NOT_TIMED:`。本工具不返回跨 host/profile 时间关系；多-host XSpace 无法安全执行 vertical context 时返回 `CLOCK_ALIGNMENT_UNKNOWN`。
- cache 损坏或身份不符以 `CACHE_READ_FAILED:` 报告；按第 4.4 节在无并发调用时执行 `mv`（优先）或仅对对应 profile hash 目录执行 `rm -r`，再用原始 `.pb` 重试。
- 禁止为了小窗口查询生成完整 Trace Viewer JSON，也不能将内部 cache serialization 暴露为用户接口。

### 5.5 使用示例

查看任意名称 XSpace 文件的结构和 cache readiness：

```bash
XPROF_CLI=/any/location/xprof-cli.py

uv run "$XPROF_CLI" \
  inspect ./capture.pb

# 总数仍完整汇报，只展示 plane_index 100 到 199
uv run "$XPROF_CLI" \
  inspect ./capture.pb --planes=100:200

# 按唯一物理 index 展示 plane 的全部 line
uv run "$XPROF_CLI" \
  inspect ./capture.pb --plane-index=1

# 只展示 line_index 100 到 199
uv run "$XPROF_CLI" \
  inspect ./capture.pb --plane-index=5 --lines=100:200
```

发现事件并执行按 plane、line、event 分组的完整统计：

```bash
uv run "$XPROF_CLI" \
  events ./capture.pb \
  --plane 'TPU*' --event '*fusion*' --match glob \
  --limit 50

uv run "$XPROF_CLI" \
  stats ./capture.pb \
  --plane 'TPU*' --event '*fusion*' --match glob \
  --group-by plane,line,event
```

根据 `events` 返回的稳定 locator 查询上下与左右文：

```bash
uv run "$XPROF_CLI" \
  context ./capture.pb \
  --event-id 'plane:0/line:3/event:42' \
  --axis both \
  --before 5 --after 5 --window-before 100us --window-after 100us \
  --include-flow
```

## 6. 验收标准

- 对官方共同语义，事件数量和 duration/aggregate 与官方结果一致。
- 同名事件不会跨 plane/line 被静默混合。
- 每条输出都能通过本次唯一输入及 plane/line/event locator 回溯到唯一原始 profile 事件；`profile_sha256` 作为独立 provenance 报告，不嵌入 event ID。
- 唯一生产文件 `xprof-cli.py` 可单独复制到任意目录；通过绝对脚本/input 路径及绝对 shell 重定向目标从任意 CWD 调用时，不读取邻近 package、仓库文件或 CWD 配置，并产生相同语义结果；cache root 始终固定为 `~/.cache/pallas-kernel/xprof-cli`。
- 所有分析命令恰好接受一个任意 basename 的原始 `*.pb`；当前只解析 XSpace 内容，其他 protobuf 格式明确拒绝。路径或文件名改变可复用缓存，内容改变不能错误命中。
- 具有可区分 wire evidence 的 `HloProto`、`OpStats`、裸 `XPlane`/`XLine` 因不能通过固定 XSpace check 而以 `INVALID_XSPACE` 拒绝；任何与合法 XSpace wire-equivalent 的歧义输入按 XSpace 解释接受，但不推断 writer 类型。`*.trace.json` 和 `*.trace.json.gz` 以 `UNSUPPORTED_INPUT_FORMAT` 拒绝；best-effort 类型猜测不能改变结果。
- cache 是可删除、可校验、可重建的加速层；cold/hot 结果语义一致，内部格式不泄漏为用户接口。
- 内部时间为整数皮秒，排序和半开区间边界有测试保证。
- `offset_ps` span/instant、`num_occurrences` aggregate 和 oneof-unset untimed 具有互斥、可测试的 kind；只有 timing-valid span/instant 进入时间统计和 context。
- 截断、近似、多-host vertical context 不可用和可选 enrichment 缺失均显式可见；不实现或暗示跨 host/profile 时钟对齐。
- context 同时覆盖跨 line 的时间重叠和同 line 的前后邻域，且不把时间重叠冒充因果 flow。
- 对大 profile 不以完整 Trace Viewer JSON 为中间格式；event、stats、context 结果保持有界，inspect 的 plane/selected-line summary 默认完整展示并可由 `--planes START:STOP`、`--lines START:STOP` 显式限制。

## 7. 相关文档

- [现有 CLI 命令参考](cli-command-reference.md)
- [XPlane protobuf parser 参考](xplane-protobuf-parser.md)
- [Trace Viewer 能力矩阵与 CLI 映射](xprof-trace-viewer-capability-matrix.md)
