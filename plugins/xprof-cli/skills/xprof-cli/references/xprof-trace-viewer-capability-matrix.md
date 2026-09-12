# XProf Trace Viewer 能力矩阵与 CLI 映射

## 1. 文档范围与审计基线

本文独立记录官方 XProf Trace Viewer 的 timeline、tracks、交互语义、数据来源和准确性边界，并说明这些能力如何映射到唯一生产文件 `scripts/xprof-cli.py`。该文件包含完整实现，可独立复制和运行，不依赖邻近 package 或仓库布局。

审计基线：

- OpenXLA/XProf `master@4f294cfac6c227daadff4676cd8fe067f37ad55e`（2026-09-01）。
- [官方 Trace Viewer 文档](https://openxla.org/xprof/trace_viewer)。
- [单 host XSpace 预处理](https://github.com/openxla/xprof/blob/4f294cfac6c227daadff4676cd8fe067f37ad55e/xprof/convert/preprocess_single_host_xplane.cc)。
- [Derived timeline 实现](https://github.com/openxla/xprof/blob/4f294cfac6c227daadff4676cd8fe067f37ad55e/xprof/utils/derived_timeline.cc)。

本文是能力和边界参考，不把 Trace Viewer UI 或完整 Trace JSON 定义为 `xprof-cli` 的内部数据模型，也不把 master 新增能力自动转换为本项目需求。项目范围、公开接口和验收标准以 [事件查询计划](xprof-cli-event-query-plan.md) 为唯一规范。

## 2. 预处理与数据来源

官方 Trace Viewer 预处理会修复或补充 HLO metadata、预处理 XPlane、生成 flow、分组，并构造 derived timelines。JAX/TF 的 name scope 可出现在派生的 `Framework Name Scope` line 中。

官方文档特别说明：只有 TPU 的 XLA Ops 和 GPU 的 stream data 被列为直接 grounded 的设备 timeline；其他 lines 可能结合 compiler sideband、用户 annotations 或 XProf heuristics 派生。因此，不能把“Trace Viewer 中看见了”自动等同于“原始 `.pb` 中存在同名 XLine”。

## 3. Sections、tracks 与数据来源

| Section/track | 平台 | 展示内容 | 数据来源 | 官方边界 | 本 CLI 对应能力 |
|---|---|---|---|---|---|
| Host CPU threads/components | CPU host | 每个 thread/threadpool 的 host events；启用时包含 Python traces。 | host tracing 与可选 Python tracing。 | Python trace 取决于采集配置；不同 host 的 clock 未必可直接比较。 | 保留 host/plane/line/thread identity，支持同 line 邻域与跨 line overlap。 |
| TPU node/chip/core section | TPU | 按 TPU node、chip/core 组织设备 timeline。 | device XPlanes 与 XProf 分组。 | section/track 命名随 TPU/XProf 版本变化。 | 不能硬编码 plane 名；用 ID、name、regex 和 capability discovery。 |
| GPU node/stream section | GPU | 每 GPU chip 下每条 stream，stream 名包含 memcpy/compute 等类别。 | GPU stream trace。 | 官方认为 GPU stream data 是直接 grounded 的主要设备真值。 | 以 stream line 为权威时序，保留并发和跨 stream overlap。 |
| SparseCore section | TPU v5p/Trillium 等 | SparseCore modules、ops 与 TraceMes。 | SparseCore trace/derived lines。 | 仅有 SparseCore 的设备与 capture 出现。 | 作为普通 plane/line 处理，不假设 dense-core schema。 |
| Steps | TPU/GPU | training/inference step 持续时间。 | framework/user step annotation 与 XProf grouping。 | 未正确标注时不出现或不完整。 | step 只作为可验证 scope/window，不反向猜测缺失 step。 |
| XLA Modules | TPU/GPU | 正在执行的编译 XLA program/module。 | runtime event 加 compiler metadata 后派生。 | module 名和 program ID sideband 可能缺失。 | 建立 module-to-event 索引并保留 provenance。 |
| TPU XLA Ops | TPU | TPU core 上执行的 HLO operations。 | TPU profile 中直接采集的 XLA op timeline。 | 官方列为直接 grounded line；仍需 HLO metadata 才有丰富详情。 | 作为 device-op 时序权威来源，关联 HLO/name scope。 |
| GPU XLA Ops | GPU | 从 kernels/streams 映射出的 HLO operations。 | 由 GPU stream data 派生。 | 可能不准确：HLO 与 kernels 是 N:M，且多 stream 会动态调度到 SM。 | 显式标记 `derived`，不能与 raw stream event 等同。 |
| XLA TraceMe | TPU | 用户、XLA 或 XProf 添加的逻辑区间，如 barrier/dropped entries。 | annotation 与派生事件。 | GPU 不支持 XLA TraceMe；即使用户未标注也可能有系统事件。 | 保留 annotation 来源，不把 TraceMe 名称当作 kernel 名。 |
| Framework Ops | TPU/GPU | 生成 XLA ops 的 JAX/TF/PyTorch framework operations。 | compiler/framework sideband 与 XProf 派生。 | 只有 framework 正确提供 annotation/metadata 时出现。 | 作为 framework-to-HLO mapping，缺失时不能猜测。 |
| Framework Name Scope | TPU/GPU | framework name-stack/scope 的 timeline；JAX 对应 `jax.named_scope`。 | derived timeline、HLO name stack 与 framework metadata。 | UI 为简洁通常只在一个 device 展示；sideband 缺失时可能不出现。 | 若该 line 已存在于输入 XSpace，则按普通 event line 查询；额外恢复属于可选 enrichment。缺失时不得让 raw-event 查询失败，也不要求为每个 JAX event 强制生成 scope 字段。 |
| Source code / Python stack | TPU/GPU/host | source path、line、Python stack 等。 | framework/compiler source metadata。 | 可选、可能含本地路径，且并非 runtime 测量。 | 可选 enrichment，记录 provenance 与隐私敏感性。 |
| Scalar unit | TPU | 在 scalar unit 执行的事件。 | TPU trace，存在时展示。 | 并非所有 TPU/capture 都有该 line。 | 作为普通设备 line 查询与统计。 |
| TensorCore Sync Flags | TPU | TPU TensorCore 同步机制事件。 | TPU runtime/trace。 | 条件出现，语义依硬件代际。 | 保留原始 stats/IDs，不用名称启发式改写。 |
| Host Offload | TPU | host↔accelerator 异步传输及对应 start/stop；并发时有多行。 | XLA ops、runtime 与派生配对。 | start/stop 和并发 row 可能依赖 sideband/后处理。 | 同时支持时间 overlap 和显式 flow/pair relation，二者分离。 |
| LLO Utilization | TPU custom call | custom call 内的底层硬件资源利用率。 | LLO debug/perf-counter metadata。 | 需要相应 XLA capture flags；缺失时不出现。 | 条件 enrichment，报告 capture readiness。 |
| GPU Launch Stats | GPU | launch phase 的 max/average latency。 | host launch 与 GPU stream correlation。 | 依赖 launch correlation 数据。 | 提供 launch→kernel flow 与统计，但不靠时间接近冒充因果。 |

## 4. Timeline 交互与查询语义

| Trace Viewer 能力 | 官方行为 | 数据/结果 | 范围边界 | 本 CLI 对应设计 |
|---|---|---|---|---|
| 单事件选择 | 点击 event，在 details pane 查看 name、start、duration 和可用 stats。 | event details。 | 可见字段取决于 event 与 sideband。 | `events` 返回稳定 locator；`context` 返回目标完整记录。 |
| 多事件选择 | Ctrl+click 多选并查看摘要。 | 选中集合 summary。 | UI 选择是交互状态，不是稳定查询。 | locator 列表输入，输出确定性 group stats。 |
| Pan/zoom/fit | W/S/A/D、工具栏和 `f` 聚焦选择。 | 改变可见时间/track 窗口。 | 不改变底层数据。 | 显式 `start/end/window-before/window-after`，结果可复现。 |
| Timing/area selection | 拖拽时间区间或 `m` 测量总 duration。 | 选定 window 的 wall time。 | wall duration 不等于其中事件 duration 总和。 | 同时报告 wall span、event sum、interval union、coverage。 |
| Streaming | 大 trace 按 pan/zoom 按需加载；未加载完成时可能先显示低分辨率数据。 | bounded viewport chunks。 | 当前 viewport 不代表全 trace。 | cache 中建立索引并分页；统计默认完整扫描且显式声明完整性。 |
| Trace Viewer v2 | 用 WebGPU/Canvas 浏览百万事件；当前只能从 UI 切换。 | 前端渲染优化。 | 不是新的 profile 语义或 CLI 数据格式。 | 不复刻前端；复用其“按窗口取数”思想。 |
| Find events | 按 event name 搜索。 | 匹配可见事件。 | 官方文档说明只搜索屏幕当前可见时间窗口，而非整个 trace。 | 默认搜索完整缓存；若限定 window，metadata 明确显示窗口。 |
| Flow Events | 用箭头连接不同 thread/line 的相关事件，如 host launch→accelerator execution。 | causal/correlation edges。 | 来自 annotations、heuristics、CUPTI/launch IDs、TPU runtime 等后处理；并非仅靠时间重叠。 | `flow_related` 与 `temporal_overlap` 分开存储和输出，并记录 provenance/confidence。 |
| XLA op details | 可显示 framework op、source/Python stack、FLOPs、bytes 等。 | runtime timing + compiler static metadata。 | FLOPs/bytes 是编译期静态信息，不是 profile runtime measurement。 | 每字段记录 `runtime/static/derived` provenance。 |
| Trace→HLO Graph | 从 XLA op detail 跳到 Graph Viewer 中对应 HLO node。 | event↔HLO node link。 | 需要可用 HloProto/program ID/op metadata。 | 输出 module/instruction locator，供 HLO 子命令继续查询。 |
| Event color | 彩色矩形区分事件。 | UI presentation。 | 官方说明颜色没有固定语义。 | 不把颜色缓存成分析字段或类别依据。 |

## 5. 对 `xprof-cli` 的硬约束

- 直接读取 XSpace 并构造可持久化的轻量索引，不把完整 Trace Viewer JSON 当作中间格式。
- derived data 只作为带 provenance 的可选 enrichment；首个里程碑不要求重建 `Framework Name Scope`，也不以其缺失判定 raw profile 无效。
- `temporal_overlap` 与 `flow_related` 分开计算和输出；时间相交不能证明因果关系。
- CLI 默认搜索完整缓存；任何 viewport/window 限制都必须在结果 metadata 中显式体现。
- runtime timing、compiler static metrics 和 XProf-derived fields 必须分别标记来源。
- 大 profile 使用分页、时间索引和 bounded output；不能用 UI 当前可见范围代替完整统计。
