#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "protobuf>=5.29.5,<6.0.0",
#   "grpcio-tools==1.71.0",
# ]
# ///
"""Portable, standalone SQLite-backed XSpace profile query CLI.

This is the complete production implementation. It resolves no files relative
to its own location: runtime behavior depends only on command-line arguments,
the input ``.pb`` file, the fixed per-user cache, and installed Python
dependencies.
"""

# 中文导读：
# - 本脚本是一个可独立复制的 XSpace 查询工具，不依赖仓库内的其他 Python 模块。
# - 原始 PB 始终按只读输入处理；解析结果按内容哈希缓存在私有 SQLite 文件中。
# - `inspect/events/stats/context` 四个命令共享同一套规范化记录和稳定事件定位符。
# - 时间统一使用整数皮秒，避免浮点换算破坏排序、区间关系和聚合结果。

from __future__ import annotations

import argparse
import base64
import binascii
import bisect
import collections
import contextlib
import fnmatch
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import sqlite3
import stat as stat_module
import struct
import sys
import tempfile
import uuid
from argparse import Namespace
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field as dataclass_field, replace
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from google.protobuf.message import DecodeError


# ============================================================================
# Constants and diagnostics
# ============================================================================

# 这些版本号分别约束公共 JSON、缓存布局、校验规则和上游 proto。
# 修改其中任意一项都可能涉及兼容性或缓存迁移，不能只把它们当普通标签更新。

CLI_SCHEMA_VERSION = "xprof-cli.v3"
CACHE_SCHEMA_VERSION = "xprof-cli-cache.v3"
CHECK_VERSION = "xspace-check.v2"
XPROF_VERSION = "2.23.1"
PARSER_MESSAGE = "tensorflow.profiler.XSpace"
PARSER_SCHEMA_REVISION = "openxla/xla@c520e3fb3f00ce8330d5088c08ee1a6f6067339f"
PARSER_SCHEMA_SHA256 = (
  "25a5097f4a62c208ea2795c9b3cc1dc2919253fefb4469f4cd2496796124493c"
)


# ============================================================================
# Pinned protobuf acquisition
# ============================================================================

# 仓库不内置生成后的 protobuf binding。首次缓存未命中时下载固定 revision 的
# xplane.proto，校验 SHA-256 后在缓存目录编译；后续命中可完全离线复用。
# schema 源文件、生成代码和 descriptor 三层哈希共同防止版本漂移或缓存污染。

SCHEMA_SOURCE_URL = (
  "https://raw.githubusercontent.com/openxla/xla/"
  "c520e3fb3f00ce8330d5088c08ee1a6f6067339f/"
  "third_party/tsl/tsl/profiler/protobuf/xplane.proto"
)
BINDING_SOURCE_SHA256 = (
  "fe6f3ec433dfd757ef19a1d65a769d6fdf05375d609d95e8609a25c9115fa977"
)
BINDING_DESCRIPTOR_SHA256 = (
  "876b5392ec3b9b641c73ff24072a570c3a3c406e7d9db4b8f8e2c46633f5d164"
)

_SCHEMA_FILENAME = "xplane.proto"
_BINDING_FILENAME = "xplane_pb2.py"
_MODULE_NAME = "_xprof_cli_runtime_xplane_pb2"
_MAX_SCHEMA_BYTES = 1024 * 1024


class CLIError(RuntimeError):
  """Expected command failure rendered as one stderr line."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(f"{code}: {' '.join(message.splitlines())}")


def diagnostic(
  code: str,
  message: str,
  *,
  severity: str = "warning",
  location: str | None = None,
  **details: Any,
) -> dict[str, Any]:
  result: dict[str, Any] = {
    "code": code,
    "severity": severity,
    "message": message,
  }
  if location is not None:
    result["location"] = location
  if details:
    result["details"] = details
  return result


# ============================================================================
# Normalized records
# ============================================================================

# 公共输出优先保持有界：长字符串改为摘要、列表限制元素数；诊断信息另有深度限制，
# 防止畸形或超大 profile 把常见查询扩张成无界 JSON。

_PUBLIC_TEXT_LIMIT = 512
_PUBLIC_LIST_LIMIT = 100


def _truncate_text(value: str, limit: int = _PUBLIC_TEXT_LIMIT) -> str:
  return value if len(value) <= limit else value[: limit - 3] + "..."


def _bounded_value(value: Any) -> Any:
  # 保留标量语义；递归处理容器、限制列表宽度，并为长文本留下可核验摘要。
  if isinstance(value, str):
    if len(value) <= _PUBLIC_TEXT_LIMIT:
      return value
    encoded = value.encode("utf-8", errors="surrogatepass")
    return {
      "truncated": True,
      "size_bytes": len(encoded),
      "sha256": hashlib.sha256(encoded).hexdigest(),
      "prefix": value[: _PUBLIC_TEXT_LIMIT - 3] + "...",
    }
  if isinstance(value, dict):
    return {
      _truncate_text(str(key)): _bounded_value(item) for key, item in value.items()
    }
  if isinstance(value, (list, tuple)):
    return [_bounded_value(item) for item in value[:_PUBLIC_LIST_LIMIT]]

  return value


@dataclass(frozen=True, slots=True)
class TypedStat:
  """One XStat with its original oneof case and metadata provenance."""

  metadata_id: int
  name: str
  description: str
  value_type: str
  value: Any
  origin: str
  ref_id: int | None = None
  ref_resolved: bool | None = None

  def to_dict(self) -> dict[str, Any]:
    result: dict[str, Any] = {
      "metadata_id": self.metadata_id,
      "name": _truncate_text(self.name),
      "description": _truncate_text(self.description),
      "value_type": self.value_type,
      "value": _bounded_value(self.value),
      "origin": self.origin,
    }
    result.update(
      {
        "name_truncated": len(self.name) > _PUBLIC_TEXT_LIMIT,
        "description_truncated": len(self.description) > _PUBLIC_TEXT_LIMIT,
      }
    )
    if self.ref_id is not None:
      result["ref_id"] = self.ref_id
      result["ref_resolved"] = self.ref_resolved
    return result


@dataclass(frozen=True, slots=True)
class EventRecord:
  """A stable normalized XEvent record with integer-picosecond timing."""

  profile_sha256: str
  source_path: str
  plane_index: int
  plane_id: int
  plane_name: str
  line_index: int
  line_id: int
  line_name: str
  line_display_name: str
  line_timestamp_ns: int
  event_ordinal: int
  metadata_id: int
  name: str
  display_name: str
  start_ps: int | None
  end_ps: int | None
  duration_ps: int
  num_occurrences: int | None
  data_case: str | None
  raw_offset_ps: int | None
  kind: str
  timing_valid: bool
  timing_unavailable_reason: str | None
  stats: tuple[TypedStat, ...] = ()
  hlo: dict[str, Any] | None = None
  flow: tuple[dict[str, Any], ...] = ()
  source_info: dict[str, Any] | None = None
  diagnostics: tuple[dict[str, Any], ...] = ()

  @property
  def event_id(self) -> str:
    # 调用已经绑定唯一 PB；定位符只描述该 XSpace 内的 protobuf 物理位置。
    return f"plane:{self.plane_index}/line:{self.line_index}/event:{self.event_ordinal}"

  @property
  def is_timed(self) -> bool:
    return self.timing_valid and self.kind in {"span", "instant"}

  def to_dict(self) -> dict[str, Any]:
    stats = self.stats[:_PUBLIC_LIST_LIMIT]
    flow = self.flow[:_PUBLIC_LIST_LIMIT]
    diagnostics = self.diagnostics[:_PUBLIC_LIST_LIMIT]
    result: dict[str, Any] = {
      "event_id": self.event_id,
      "profile_sha256": self.profile_sha256,
      "source_path": self.source_path,
      "plane_index": self.plane_index,
      "plane_id": self.plane_id,
      "plane_name": _truncate_text(self.plane_name),
      "line_index": self.line_index,
      "line_id": self.line_id,
      "line_name": _truncate_text(self.line_name),
      "line_display_name": _truncate_text(self.line_display_name),
      "line_timestamp_ns": self.line_timestamp_ns,
      "event_ordinal": self.event_ordinal,
      "metadata_id": self.metadata_id,
      "name": _truncate_text(self.name),
      "display_name": _truncate_text(self.display_name),
      "start_ps": self.start_ps,
      "end_ps": self.end_ps,
      "duration_ps": self.duration_ps,
      "num_occurrences": self.num_occurrences,
      "data_case": self.data_case,
      "raw_offset_ps": self.raw_offset_ps,
      "kind": self.kind,
      "timing_valid": self.timing_valid,
      "timing_unavailable_reason": self.timing_unavailable_reason,
      "stats": [stat.to_dict() for stat in stats],
      "hlo": _bounded_value(self.hlo),
      "flow": _bounded_value(flow),
      "source_info": _bounded_value(self.source_info),
      "diagnostics": _bounded_value(diagnostics),
    }
    result.update(
      {
        "plane_name_truncated": len(self.plane_name) > _PUBLIC_TEXT_LIMIT,
        "line_name_truncated": len(self.line_name) > _PUBLIC_TEXT_LIMIT,
        "line_display_name_truncated": (
          len(self.line_display_name) > _PUBLIC_TEXT_LIMIT
        ),
        "name_truncated": len(self.name) > _PUBLIC_TEXT_LIMIT,
        "display_name_truncated": len(self.display_name) > _PUBLIC_TEXT_LIMIT,
        "stat_count": len(self.stats),
        "stats_returned_count": len(stats),
        "stats_truncated": len(stats) < len(self.stats),
        "flow_count": len(self.flow),
        "flow_returned_count": len(flow),
        "flow_truncated": len(flow) < len(self.flow),
        "diagnostic_count": len(self.diagnostics),
        "diagnostics_returned_count": len(diagnostics),
        "diagnostics_truncated": len(diagnostics) < len(self.diagnostics),
      }
    )
    return result


@dataclass(frozen=True, slots=True)
class ProfileRecord:
  """Invocation binding for one immutable SQLite profile."""

  profile_sha256: str
  source_path: str
  raw_size_bytes: int
  database_path: str = dataclass_field(repr=False)
  cache: dict[str, Any] = dataclass_field(default_factory=dict)


# ============================================================================
# Pinned protobuf acquisition
# ============================================================================


def _prepare_directory(path: Path) -> None:
  try:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat_module.S_ISDIR(metadata.st_mode):
      raise OSError("schema cache path is not a directory")
    os.chmod(path, 0o700)
  except OSError as error:
    raise CLIError(
      "SCHEMA_CACHE_FAILED",
      f"failed to prepare protobuf schema cache {path}: {error}",
    ) from error


def _matches(path: Path, expected_sha256: str) -> bool:
  try:
    metadata = path.lstat()
    if not stat_module.S_ISREG(metadata.st_mode):
      return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
      while chunk := source.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest() == expected_sha256
  except FileNotFoundError:
    return False
  except OSError as error:
    raise CLIError(
      "SCHEMA_CACHE_FAILED",
      f"failed to inspect cached protobuf artifact {path}: {error}",
    ) from error


def _publish(path: Path, payload: bytes) -> None:
  # 先写独占临时文件并 fsync，再原子替换目标；这样并发或崩溃不会留下半个文件。
  temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
  descriptor: int | None = None
  try:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as output:
      descriptor = None
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(
      path.parent,
      os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  except OSError as error:
    if descriptor is not None:
      os.close(descriptor)
    with contextlib.suppress(FileNotFoundError):
      temporary.unlink()
    raise CLIError(
      "SCHEMA_CACHE_FAILED",
      f"failed to publish protobuf schema artifact {path}: {error}",
    ) from error


def _download_schema(path: Path) -> None:
  # 下载量设置 1 MiB 硬上限，且任何内容都必须匹配固定上游文件的摘要。
  request = Request(
    SCHEMA_SOURCE_URL,
    headers={"User-Agent": "pallas-kernel-xprof-cli/1"},
  )
  try:
    payload = bytearray()
    with urlopen(request, timeout=30) as response:
      while chunk := response.read(64 * 1024):
        payload.extend(chunk)
        if len(payload) > _MAX_SCHEMA_BYTES:
          raise ValueError("downloaded schema exceeds the 1 MiB limit")
  except (OSError, URLError, ValueError) as error:
    raise CLIError(
      "SCHEMA_DOWNLOAD_FAILED",
      f"failed to download pinned XPlane schema from {SCHEMA_SOURCE_URL} "
      f"to {path}: {error}",
    ) from error
  observed = hashlib.sha256(payload).hexdigest()
  if observed != PARSER_SCHEMA_SHA256:
    raise CLIError(
      "SCHEMA_DOWNLOAD_FAILED",
      f"downloaded XPlane schema at {path} failed SHA-256 verification",
    )
  _publish(path, bytes(payload))


def _compile_binding(schema_path: Path, binding_path: Path) -> None:
  # 在一次性目录编译，只有生成物哈希完全匹配预期时才发布到共享缓存。
  try:
    from grpc_tools import protoc
  except ImportError as error:
    raise CLIError(
      "SCHEMA_COMPILER_UNAVAILABLE",
      "grpcio-tools 1.71.0 is required to compile the downloaded XPlane schema",
    ) from error

  build_dir = Path(tempfile.mkdtemp(prefix=".xplane-binding-", dir=schema_path.parent))
  try:
    result = protoc.main(
      [
        "grpc_tools.protoc",
        f"-I{schema_path.parent}",
        f"--python_out={build_dir}",
        str(schema_path),
      ]
    )
    candidate = build_dir / _BINDING_FILENAME
    if result != 0 or not _matches(candidate, BINDING_SOURCE_SHA256):
      raise CLIError(
        "SCHEMA_COMPILE_FAILED",
        "the pinned XPlane schema did not produce the expected Python binding "
        f"(protoc exit {result})",
      )
    _publish(binding_path, candidate.read_bytes())
  finally:
    shutil.rmtree(build_dir, ignore_errors=True)


def _import_binding(source: bytes, path: Path) -> ModuleType:
  # `exec` 的对象是已校验的固定生成物；导入后再校验 descriptor，形成第二道边界。
  existing = sys.modules.get(_MODULE_NAME)
  try:
    if existing is None:
      module = ModuleType(_MODULE_NAME)
      module.__file__ = str(path)
      exec(compile(source, str(path), "exec"), module.__dict__)
    else:
      module = existing
    descriptor_hash = hashlib.sha256(module.DESCRIPTOR.serialized_pb).hexdigest()
    if descriptor_hash != BINDING_DESCRIPTOR_SHA256:
      raise ImportError("generated descriptor SHA-256 does not match")
    if module.XSpace.DESCRIPTOR.full_name != "tensorflow.profiler.XSpace":
      raise ImportError("generated binding does not define tensorflow.profiler.XSpace")
  except (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
  ) as error:
    raise CLIError(
      "SCHEMA_BINDING_FAILED",
      f"failed to load cached XPlane binding {path}: {error}",
    ) from error
  sys.modules[_MODULE_NAME] = module
  return module


def load_xplane_binding(schema_dir: Path) -> ModuleType:
  """Download, compile, and load the pinned schema from the SQLite cache root."""
  # 每一步都重新核对哈希，避免“检查后、读取前”缓存内容被替换。
  schema_dir = Path(schema_dir)
  _prepare_directory(schema_dir)
  schema_path = schema_dir / _SCHEMA_FILENAME
  binding_path = schema_dir / _BINDING_FILENAME
  if not _matches(schema_path, PARSER_SCHEMA_SHA256):
    _download_schema(schema_path)
  if not _matches(binding_path, BINDING_SOURCE_SHA256):
    _compile_binding(schema_path, binding_path)
  try:
    binding_source = binding_path.read_bytes()
  except OSError as error:
    raise CLIError(
      "SCHEMA_BINDING_FAILED",
      f"failed to read cached XPlane binding {binding_path}: {error}",
    ) from error
  if hashlib.sha256(binding_source).hexdigest() != BINDING_SOURCE_SHA256:
    raise CLIError(
      "SCHEMA_BINDING_FAILED",
      f"cached XPlane binding {binding_path} changed after SHA-256 verification",
    )
  return _import_binding(binding_source, binding_path)


# ============================================================================
# XSpace parsing
# ============================================================================

# 解析层只把原始 protobuf 规范化为稳定字段，不猜测缺失语义。
# HLO、flow 和源码位置仅来自类型正确且无冲突的原始 XStat；无法证明时就省略。

_HLO_KEYS = {
  "hlo_op": "op",
  "hlo_op_name": "op",
  "hlo_module": "module",
  "hlo_module_name": "module",
  "hlo_category": "category",
}
_FLOW_KEYS = {
  "flow",
  "flow_id",
  "flowid",
  "correlation_id",
  "correlationid",
}
_SOURCE_KEYS = {
  "source_file": "file",
  "source_filename": "file",
  "source_line": "line",
  "source_column": "column",
  "source_function": "function",
}
_PARSER_INT64_MIN = -(2**63)
_PARSER_INT64_MAX = 2**63 - 1


def validate_input_path(path: Path) -> Path:
  """Validate the public raw-profile path contract without modifying it."""
  # 这里只接受普通 `.pb` 文件；目录、压缩包和 JSON 不做隐式解包或格式猜测。
  try:
    resolved = path.expanduser().resolve()
  except (OSError, RuntimeError) as error:
    raise CLIError(
      "UNSUPPORTED_INPUT_FORMAT",
      f"PATH_RESOLUTION_FAILED: profile path {path} could not be resolved: {error}",
    ) from error
  if not resolved.exists():
    raise CLIError("FILE_NOT_FOUND", f"profile file does not exist: {resolved}")
  if not resolved.is_file():
    raise CLIError(
      "UNSUPPORTED_INPUT_FORMAT",
      f"NOT_A_REGULAR_FILE: profile path is not a regular file: {resolved}",
    )
  if resolved.suffix.lower() != ".pb":
    raise CLIError(
      "UNSUPPORTED_INPUT_FORMAT",
      f"FILE_EXTENSION_NOT_PB: only raw .pb files are accepted: {resolved}",
    )
  return resolved


def _looks_like_non_protobuf(raw: bytes | bytearray | memoryview) -> str | None:
  prefix = bytes(raw[:256])
  if prefix.startswith(b"\x1f\x8b"):
    return "GZIP_DATA"
  prefix = prefix.lstrip()
  if prefix.startswith((b"{", b"[")):
    return "JSON_DATA"
  return None


def _location(
  plane_index: int, line_index: int | None = None, event: int | None = None
) -> str:
  result = f"plane:{plane_index}"
  if line_index is not None:
    result += f"/line:{line_index}"
  if event is not None:
    result += f"/event:{event}"
  return result


def _normal_name(name: str) -> str:
  return "_".join(name.strip().lower().replace("/", " ").split())


def _stat_value(
  stat: Any,
  *,
  plane: Any,
  origin: str,
  location: str,
  diagnostics: list[dict[str, Any]],
) -> TypedStat:
  # 严格保留 XStat oneof 类型；非有限浮点和未解析引用不会被伪装成普通数值。
  metadata = plane.stat_metadata.get(stat.metadata_id)
  if metadata is None:
    name = f"<stat-metadata:{stat.metadata_id}>"
    description = ""
    diagnostics.append(
      diagnostic(
        "STAT_METADATA_NOT_FOUND",
        "XStat metadata_id is absent from its XPlane metadata map",
        location=location,
        metadata_id=stat.metadata_id,
      )
    )
  else:
    name = metadata.name or f"<stat-metadata:{stat.metadata_id}>"
    description = metadata.description

  value_case = stat.WhichOneof("value")
  if value_case is None:
    diagnostics.append(
      diagnostic(
        "STAT_VALUE_UNSET",
        "XStat value oneof is unset",
        location=location,
        metadata_id=stat.metadata_id,
      )
    )
    return TypedStat(
      metadata_id=stat.metadata_id,
      name=name,
      description=description,
      value_type="unset",
      value=None,
      origin=origin,
    )

  if value_case == "double_value":
    value = stat.double_value
    if not math.isfinite(value):
      rendered = (
        "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
      )
      value = {"non_finite": rendered}
      diagnostics.append(
        diagnostic(
          "NON_FINITE_STAT_VALUE",
          "non-finite double XStat is preserved but excluded from numeric aggregation",
          location=location,
          metadata_id=stat.metadata_id,
          value=rendered,
        )
      )
    return TypedStat(stat.metadata_id, name, description, "double", value, origin)
  if value_case == "uint64_value":
    return TypedStat(
      stat.metadata_id,
      name,
      description,
      "uint64",
      int(stat.uint64_value),
      origin,
    )
  if value_case == "int64_value":
    return TypedStat(
      stat.metadata_id,
      name,
      description,
      "int64",
      int(stat.int64_value),
      origin,
    )
  if value_case == "str_value":
    return TypedStat(
      stat.metadata_id, name, description, "string", stat.str_value, origin
    )
  if value_case == "bytes_value":
    raw_value = bytes(stat.bytes_value)
    summary = {
      "size_bytes": len(raw_value),
      "sha256": hashlib.sha256(raw_value).hexdigest(),
      "hex_prefix": raw_value[:16].hex(),
    }
    return TypedStat(
      stat.metadata_id,
      name,
      description,
      "bytes",
      summary,
      origin,
    )

  ref_id = int(stat.ref_value)
  ref_metadata = (
    plane.stat_metadata.get(ref_id)
    if _PARSER_INT64_MIN <= ref_id <= _PARSER_INT64_MAX
    else None
  )
  resolved = ref_metadata is not None
  if not resolved:
    diagnostics.append(
      diagnostic(
        "STAT_REF_NOT_FOUND",
        "XStat ref_value is absent from its XPlane stat metadata map",
        location=location,
        metadata_id=stat.metadata_id,
        ref_id=ref_id,
      )
    )
  return TypedStat(
    stat.metadata_id,
    name,
    description,
    "ref",
    ref_metadata.name if resolved else None,
    origin,
    ref_id=ref_id,
    ref_resolved=resolved,
  )


def _normalize_stats(
  stats: Iterable[Any],
  *,
  plane: Any,
  origin: str,
  location: str,
  diagnostics: list[dict[str, Any]],
) -> list[TypedStat]:
  # 同一作用域的重复 metadata_id 仍全部保留，但附加诊断，避免静默丢数据。
  result: list[TypedStat] = []
  seen: set[int] = set()
  for index, stat in enumerate(stats):
    stat_location = f"{location}/stat:{index}"
    if stat.metadata_id in seen:
      diagnostics.append(
        diagnostic(
          "DUPLICATE_STAT_METADATA_ID",
          "multiple XStats in the same scope use one metadata_id; all values are preserved",
          location=stat_location,
          metadata_id=stat.metadata_id,
          origin=origin,
        )
      )
    seen.add(stat.metadata_id)
    result.append(
      _stat_value(
        stat,
        plane=plane,
        origin=origin,
        location=stat_location,
        diagnostics=diagnostics,
      )
    )
  return result


def _enrich(
  stats: Iterable[TypedStat],
  *,
  location: str,
  diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...], dict[str, Any] | None]:
  # enrichment 是保守映射：同一目标字段有冲突或类型不符时，宁可不输出。
  hlo: dict[str, Any] = {}
  flow: list[dict[str, Any]] = []
  source: dict[str, Any] = {}
  hlo_seen: dict[str, tuple[str, Any]] = {}
  source_seen: dict[str, tuple[str, Any]] = {}
  ambiguous_hlo: set[str] = set()
  ambiguous_source: set[str] = set()
  invalid_hlo_types: set[str] = set()
  invalid_source_types: set[str] = set()
  for stat in stats:
    key = _normal_name(stat.name)
    scalar = stat.value
    if stat.value_type == "bytes" or scalar is None or isinstance(scalar, dict):
      continue
    if key in _HLO_KEYS:
      canonical = _HLO_KEYS[key]
      if stat.value_type not in {"string", "ref"} or not isinstance(scalar, str):
        invalid_hlo_types.add(canonical)
      else:
        identity = (stat.value_type, scalar)
        previous = hlo_seen.setdefault(canonical, identity)
        if previous == identity and canonical not in ambiguous_hlo:
          hlo[canonical] = scalar
        elif previous != identity:
          ambiguous_hlo.add(canonical)
          hlo.pop(canonical, None)
    if key in _FLOW_KEYS:
      flow.append(
        {
          "name": stat.name,
          "value_type": stat.value_type,
          "value": scalar,
          "provenance": "raw_xstat",
        }
      )
    if key in _SOURCE_KEYS:
      canonical = _SOURCE_KEYS[key]
      valid_type = (
        canonical in {"file", "function"}
        and stat.value_type in {"string", "ref"}
        and isinstance(scalar, str)
        or canonical in {"line", "column"}
        and stat.value_type in {"int64", "uint64"}
        and isinstance(scalar, int)
        and not isinstance(scalar, bool)
      )
      if not valid_type:
        invalid_source_types.add(canonical)
      else:
        identity = (stat.value_type, scalar)
        previous = source_seen.setdefault(canonical, identity)
        if previous == identity and canonical not in ambiguous_source:
          source[canonical] = scalar
        elif previous != identity:
          ambiguous_source.add(canonical)
          source.pop(canonical, None)
  if ambiguous_hlo:
    diagnostics.append(
      diagnostic(
        "AMBIGUOUS_HLO_ENRICHMENT",
        "conflicting raw XStats mapped to the same HLO field; ambiguous fields were omitted",
        location=location,
        fields=sorted(ambiguous_hlo),
      )
    )
  if ambiguous_source:
    diagnostics.append(
      diagnostic(
        "AMBIGUOUS_SOURCE_ENRICHMENT",
        "conflicting raw XStats mapped to the same source field; ambiguous fields were omitted",
        location=location,
        fields=sorted(ambiguous_source),
      )
    )
  if invalid_hlo_types:
    diagnostics.append(
      diagnostic(
        "INVALID_HLO_ENRICHMENT_TYPE",
        "HLO name enrichment requires string or resolved-reference XStats; invalid fields were omitted",
        location=location,
        fields=sorted(invalid_hlo_types),
      )
    )
  if invalid_source_types:
    diagnostics.append(
      diagnostic(
        "INVALID_SOURCE_ENRICHMENT_TYPE",
        "source enrichment XStats had incompatible value types; invalid fields were omitted",
        location=location,
        fields=sorted(invalid_source_types),
      )
    )
  if hlo:
    hlo["provenance"] = "raw_xstat"
  if source:
    source["provenance"] = "raw_xstat"
  return hlo or None, tuple(flow), source or None


def _validate_metadata_maps(space: Any) -> list[dict[str, Any]]:
  # protobuf map 的 key 与消息内 id 不一致会破坏后续关联，因此视为致命结构错误。
  fatal: list[dict[str, Any]] = []
  for plane_index, plane in enumerate(space.planes):
    for key in sorted(plane.event_metadata):
      metadata = plane.event_metadata[key]
      if key != metadata.id:
        fatal.append(
          {
            "code": "EVENT_METADATA_KEY_ID_MISMATCH",
            "location": _location(plane_index),
            "map_key": key,
            "metadata_id": metadata.id,
          }
        )
    for key in sorted(plane.stat_metadata):
      metadata = plane.stat_metadata[key]
      if key != metadata.id:
        fatal.append(
          {
            "code": "STAT_METADATA_KEY_ID_MISMATCH",
            "location": _location(plane_index),
            "map_key": key,
            "metadata_id": metadata.id,
          }
        )
  return fatal


def _decode_xspace(
  raw: bytes | bytearray | memoryview,
  *,
  source_path: Path,
  profile_sha256: str,
  xplane_module: Any,
) -> tuple[Any, bool]:
  """Decode and semantically validate one complete serialized XSpace."""
  # 先排除常见误输入，再做 protobuf 解码和最小语义校验；未知字段单独标记。
  non_proto_reason = _looks_like_non_protobuf(raw)
  if non_proto_reason is not None:
    raise CLIError(
      "UNSUPPORTED_INPUT_FORMAT",
      f"{non_proto_reason}: {source_path} is recognizable non-protobuf data",
    )

  space = xplane_module.XSpace()
  try:
    space.ParseFromString(raw)
  except DecodeError as error:
    raise CLIError(
      "INVALID_XSPACE",
      f"PROTO_DECODE_FAILED: failed to decode {source_path}: {error}",
    ) from error

  if not (space.planes or space.errors or space.warnings or space.hostnames):
    raise CLIError(
      "INVALID_XSPACE",
      f"EMPTY_XSPACE: decoded input {source_path} contains no XSpace evidence",
    )

  before_unknown_discard = space.ByteSize()
  space.DiscardUnknownFields()
  unknown_fields_present = space.ByteSize() != before_unknown_discard
  fatal = _validate_metadata_maps(space)
  if fatal:
    failures = ", ".join(f"{item['code']} at {item['location']}" for item in fatal)
    raise CLIError(
      "INVALID_XSPACE",
      f"SEMANTIC_CHECK_FAILED for {source_path}: {failures}",
    )
  return space, unknown_fields_present


# ============================================================================
# SQLite cache
# ============================================================================

# 缓存是按 profile SHA-256 寻址的不可变分析索引。STRICT 表和 CHECK/FK 约束
# 同时承担持久化格式校验，查询时仍会核对 application_id、schema 版本和输入身份。
# 体量随事件数增长的状态尽量留在 SQLite/临时文件中，避免构造 profile 级 Python 列表。

SQLITE_SCHEMA_VERSION = 1
SQLITE_APPLICATION_ID = 0x58504643  # ASCII "XPFC".

_UINT64_MAX = 2**64 - 1
_I128_MIN = -(2**127)
_I128_MAX = 2**127 - 1
_KINDS = ("span", "instant", "aggregate", "untimed")


class SQLiteCacheError(ValueError):
  """The SQLite artifact or direct builder violates the private contract."""


def _canonical_json(value: Any) -> str:
  try:
    return json.dumps(
      value,
      ensure_ascii=False,
      allow_nan=False,
      sort_keys=True,
      separators=(",", ":"),
    )
  except (RecursionError, TypeError, ValueError) as error:
    raise SQLiteCacheError(
      f"value is not canonical-JSON serializable: {error}"
    ) from error


def _json_loads(raw: str) -> Any:
  def reject_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r}")

  try:
    return json.loads(raw, parse_constant=reject_constant)
  except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
    raise SQLiteCacheError(
      f"invalid canonical JSON stored in cache: {error}"
    ) from error


_TABLE_SQL: tuple[str, ...] = (
  """
    CREATE TABLE profile (
      singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
      profile_sha256 TEXT NOT NULL CHECK (length(profile_sha256) = 64),
      raw_size_bytes INTEGER NOT NULL CHECK (raw_size_bytes >= 0),
      span_count INTEGER NOT NULL CHECK (span_count >= 0),
      instant_count INTEGER NOT NULL CHECK (instant_count >= 0),
      aggregate_count INTEGER NOT NULL CHECK (aggregate_count >= 0),
      untimed_count INTEGER NOT NULL CHECK (untimed_count >= 0),
      validation_json TEXT NOT NULL
    ) STRICT
    """,
  """
    CREATE TABLE hostname (
      ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
      value TEXT NOT NULL
    ) STRICT
    """,
  """
    CREATE TABLE capture_message (
      kind TEXT NOT NULL CHECK (kind IN ('error', 'warning')),
      ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
      value TEXT NOT NULL,
      PRIMARY KEY (kind, ordinal)
    ) STRICT, WITHOUT ROWID
    """,
  """
    CREATE TABLE plane (
      plane_index INTEGER PRIMARY KEY CHECK (plane_index >= 0),
      plane_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      line_count INTEGER NOT NULL CHECK (line_count >= 0),
      event_metadata_count INTEGER NOT NULL CHECK (event_metadata_count >= 0),
      stat_metadata_count INTEGER NOT NULL CHECK (stat_metadata_count >= 0)
    ) STRICT
    """,
  """
    CREATE TABLE line (
      plane_index INTEGER NOT NULL,
      line_index INTEGER NOT NULL CHECK (line_index >= 0),
      line_id INTEGER NOT NULL,
      display_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      display_name TEXT NOT NULL,
      timestamp_ns INTEGER NOT NULL,
      duration_ps INTEGER NOT NULL,
      event_count INTEGER NOT NULL CHECK (event_count >= 0),
      span_count INTEGER NOT NULL CHECK (span_count >= 0),
      instant_count INTEGER NOT NULL CHECK (instant_count >= 0),
      aggregate_count INTEGER NOT NULL CHECK (aggregate_count >= 0),
      untimed_count INTEGER NOT NULL CHECK (untimed_count >= 0),
      PRIMARY KEY (plane_index, line_index),
      FOREIGN KEY (plane_index) REFERENCES plane(plane_index)
    ) STRICT, WITHOUT ROWID
    """,
  """
    CREATE TABLE event_metadata (
      plane_index INTEGER NOT NULL,
      metadata_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      display_name TEXT NOT NULL,
      metadata_bytes BLOB NOT NULL,
      metadata_sha256 TEXT NOT NULL CHECK (length(metadata_sha256) = 64),
      PRIMARY KEY (plane_index, metadata_id),
      FOREIGN KEY (plane_index) REFERENCES plane(plane_index)
    ) STRICT, WITHOUT ROWID
    """,
  """
    CREATE TABLE event_metadata_child (
      plane_index INTEGER NOT NULL,
      metadata_id INTEGER NOT NULL,
      ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
      child_id INTEGER NOT NULL,
      PRIMARY KEY (plane_index, metadata_id, ordinal),
      FOREIGN KEY (plane_index, metadata_id)
        REFERENCES event_metadata(plane_index, metadata_id)
    ) STRICT, WITHOUT ROWID
    """,
  """
    CREATE TABLE stat_definition (
      plane_index INTEGER NOT NULL,
      metadata_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      description TEXT NOT NULL,
      PRIMARY KEY (plane_index, metadata_id),
      FOREIGN KEY (plane_index) REFERENCES plane(plane_index)
    ) STRICT, WITHOUT ROWID
    """,
  """
    CREATE TABLE event (
      event_pk INTEGER PRIMARY KEY,
      plane_index INTEGER NOT NULL,
      line_index INTEGER NOT NULL,
      event_ordinal INTEGER NOT NULL CHECK (event_ordinal >= 0),
      metadata_id INTEGER NOT NULL,
      data_case TEXT CHECK (data_case IS NULL OR data_case IN ('offset_ps', 'num_occurrences')),
      raw_offset_ps INTEGER,
      duration_ps INTEGER NOT NULL,
      num_occurrences INTEGER,
      kind TEXT NOT NULL CHECK (kind IN ('span', 'instant', 'aggregate', 'untimed')),
      timing_valid INTEGER NOT NULL CHECK (timing_valid IN (0, 1)),
      start_ps BLOB CHECK (start_ps IS NULL OR length(start_ps) = 16),
      end_ps BLOB CHECK (end_ps IS NULL OR length(end_ps) = 16),
      timing_unavailable_reason TEXT,
      UNIQUE (plane_index, line_index, event_ordinal),
      FOREIGN KEY (plane_index, line_index) REFERENCES line(plane_index, line_index),
      CHECK (
        (kind = 'span' AND data_case = 'offset_ps' AND raw_offset_ps IS NOT NULL
          AND duration_ps > 0 AND num_occurrences IS NULL AND timing_valid = 1
          AND start_ps IS NOT NULL AND end_ps IS NOT NULL
          AND timing_unavailable_reason IS NULL)
        OR
        (kind = 'instant' AND data_case = 'offset_ps' AND raw_offset_ps IS NOT NULL
          AND duration_ps = 0 AND num_occurrences IS NULL AND timing_valid = 1
          AND start_ps IS NOT NULL AND end_ps IS NOT NULL
          AND timing_unavailable_reason IS NULL)
        OR
        (kind = 'aggregate' AND data_case = 'num_occurrences'
          AND raw_offset_ps IS NULL AND num_occurrences IS NOT NULL
          AND timing_valid = 0 AND start_ps IS NULL AND end_ps IS NULL
          AND timing_unavailable_reason = 'AGGREGATE_RECORD')
        OR
        (kind = 'untimed' AND timing_valid = 0 AND start_ps IS NULL AND end_ps IS NULL
          AND ((data_case = 'offset_ps' AND raw_offset_ps IS NOT NULL
                AND duration_ps < 0 AND num_occurrences IS NULL
                AND timing_unavailable_reason = 'NEGATIVE_DURATION')
            OR (data_case IS NULL AND raw_offset_ps IS NULL
                AND num_occurrences IS NULL
                AND timing_unavailable_reason = 'DATA_ONEOF_UNSET')))
      )
    ) STRICT
    """,
  """
    CREATE TABLE stat (
      stat_pk INTEGER PRIMARY KEY,
      owner_kind TEXT NOT NULL CHECK (owner_kind IN ('plane', 'event_metadata', 'event')),
      plane_index INTEGER NOT NULL,
      owner_plane_index INTEGER,
      owner_event_metadata_id INTEGER,
      owner_event_pk INTEGER,
      ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
      metadata_id INTEGER NOT NULL,
      value_type TEXT NOT NULL CHECK (
        value_type IN ('unset', 'double', 'uint64', 'int64', 'string', 'bytes', 'ref')
      ),
      double_bits BLOB CHECK (double_bits IS NULL OR length(double_bits) = 8),
      nonfinite_tag TEXT CHECK (
        nonfinite_tag IS NULL OR nonfinite_tag IN ('NaN', 'Infinity', '-Infinity')
      ),
      uint64_value BLOB CHECK (uint64_value IS NULL OR length(uint64_value) = 8),
      int64_value INTEGER,
      string_value TEXT,
      bytes_value BLOB,
      bytes_sha256 TEXT CHECK (bytes_sha256 IS NULL OR length(bytes_sha256) = 64),
      ref_id BLOB CHECK (ref_id IS NULL OR length(ref_id) = 8),
      ref_value TEXT,
      ref_resolved INTEGER CHECK (ref_resolved IS NULL OR ref_resolved IN (0, 1)),
      FOREIGN KEY (plane_index, metadata_id)
        REFERENCES stat_definition(plane_index, metadata_id),
      FOREIGN KEY (owner_plane_index) REFERENCES plane(plane_index),
      FOREIGN KEY (plane_index, owner_event_metadata_id)
        REFERENCES event_metadata(plane_index, metadata_id),
      FOREIGN KEY (owner_event_pk) REFERENCES event(event_pk),
      CHECK (
        (owner_kind = 'plane' AND owner_plane_index = plane_index
          AND owner_event_metadata_id IS NULL AND owner_event_pk IS NULL)
        OR
        (owner_kind = 'event_metadata' AND owner_plane_index IS NULL
          AND owner_event_metadata_id IS NOT NULL AND owner_event_pk IS NULL)
        OR
        (owner_kind = 'event' AND owner_plane_index IS NULL
          AND owner_event_metadata_id IS NULL AND owner_event_pk IS NOT NULL)
      ),
      CHECK (
        (value_type = 'unset' AND double_bits IS NULL AND nonfinite_tag IS NULL
          AND uint64_value IS NULL AND int64_value IS NULL AND string_value IS NULL
          AND bytes_value IS NULL AND bytes_sha256 IS NULL AND ref_id IS NULL
          AND ref_value IS NULL AND ref_resolved IS NULL)
        OR
        (value_type = 'double' AND double_bits IS NOT NULL
          AND uint64_value IS NULL AND int64_value IS NULL AND string_value IS NULL
          AND bytes_value IS NULL AND bytes_sha256 IS NULL AND ref_id IS NULL
          AND ref_value IS NULL AND ref_resolved IS NULL)
        OR
        (value_type = 'uint64' AND double_bits IS NULL AND nonfinite_tag IS NULL
          AND uint64_value IS NOT NULL AND int64_value IS NULL AND string_value IS NULL
          AND bytes_value IS NULL AND bytes_sha256 IS NULL AND ref_id IS NULL
          AND ref_value IS NULL AND ref_resolved IS NULL)
        OR
        (value_type = 'int64' AND double_bits IS NULL AND nonfinite_tag IS NULL
          AND uint64_value IS NULL AND int64_value IS NOT NULL AND string_value IS NULL
          AND bytes_value IS NULL AND bytes_sha256 IS NULL AND ref_id IS NULL
          AND ref_value IS NULL AND ref_resolved IS NULL)
        OR
        (value_type = 'string' AND double_bits IS NULL AND nonfinite_tag IS NULL
          AND uint64_value IS NULL AND int64_value IS NULL AND string_value IS NOT NULL
          AND bytes_value IS NULL AND bytes_sha256 IS NULL AND ref_id IS NULL
          AND ref_value IS NULL AND ref_resolved IS NULL)
        OR
        (value_type = 'bytes' AND double_bits IS NULL AND nonfinite_tag IS NULL
          AND uint64_value IS NULL AND int64_value IS NULL AND string_value IS NULL
          AND bytes_value IS NOT NULL AND bytes_sha256 IS NOT NULL AND ref_id IS NULL
          AND ref_value IS NULL AND ref_resolved IS NULL)
        OR
        (value_type = 'ref' AND double_bits IS NULL AND nonfinite_tag IS NULL
          AND uint64_value IS NULL AND int64_value IS NULL AND string_value IS NULL
          AND bytes_value IS NULL AND bytes_sha256 IS NULL AND ref_id IS NOT NULL
          AND ref_resolved IS NOT NULL)
      )
    ) STRICT
    """,
  """
    CREATE TABLE event_enrichment (
      event_pk INTEGER PRIMARY KEY,
      hlo_op TEXT,
      hlo_json TEXT,
      source_info_json TEXT,
      CHECK (hlo_json IS NOT NULL OR source_info_json IS NOT NULL),
      FOREIGN KEY (event_pk) REFERENCES event(event_pk)
    ) STRICT
    """,
  """
    CREATE TABLE event_flow (
      event_pk INTEGER NOT NULL,
      ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
      flow_key TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      PRIMARY KEY (event_pk, ordinal),
      FOREIGN KEY (event_pk) REFERENCES event(event_pk)
    ) STRICT, WITHOUT ROWID
    """,
  """
    CREATE TABLE diagnostic (
      ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
      code TEXT NOT NULL,
      message TEXT NOT NULL,
      details_json TEXT NOT NULL
    ) STRICT
    """,
  """
    CREATE TABLE event_diagnostic (
      event_pk INTEGER NOT NULL,
      ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
      diagnostic_ordinal INTEGER NOT NULL,
      PRIMARY KEY (event_pk, ordinal),
      UNIQUE (event_pk, diagnostic_ordinal),
      FOREIGN KEY (event_pk) REFERENCES event(event_pk),
      FOREIGN KEY (diagnostic_ordinal) REFERENCES diagnostic(ordinal)
    ) STRICT, WITHOUT ROWID
    """,
)
_INDEX_SQL: tuple[str, ...] = (
  "CREATE INDEX idx_line_logical ON line(plane_index, line_id, line_index)",
  "CREATE INDEX idx_event_metadata_name ON event_metadata(name, plane_index, metadata_id)",
  "CREATE INDEX idx_event_metadata_display_name ON event_metadata(display_name, plane_index, metadata_id)",
  "CREATE INDEX idx_event_by_metadata ON event(plane_index, metadata_id, line_index, event_ordinal)",
  "CREATE INDEX idx_event_by_time ON event(start_ps, end_ps, plane_index, line_index, event_ordinal) WHERE timing_valid = 1",
  "CREATE INDEX idx_event_by_duration ON event(duration_ps, plane_index, line_index, event_ordinal)",
  "CREATE INDEX idx_event_by_kind ON event(kind, plane_index, line_index, event_ordinal)",
  "CREATE INDEX idx_stat_definition_name ON stat_definition(name, plane_index, metadata_id)",
  "CREATE UNIQUE INDEX idx_stat_plane_owner ON stat(owner_plane_index, ordinal) WHERE owner_kind = 'plane'",
  "CREATE UNIQUE INDEX idx_stat_event_metadata_owner ON stat(plane_index, owner_event_metadata_id, ordinal) WHERE owner_kind = 'event_metadata'",
  "CREATE UNIQUE INDEX idx_stat_event_owner ON stat(owner_event_pk, ordinal) WHERE owner_kind = 'event'",
  "CREATE INDEX idx_stat_metadata_value ON stat(plane_index, metadata_id, value_type)",
  "CREATE INDEX idx_event_enrichment_hlo_op ON event_enrichment(hlo_op, event_pk) WHERE hlo_op IS NOT NULL",
  "CREATE INDEX idx_event_flow_key ON event_flow(flow_key, event_pk)",
)


def encode_ordered_i128(value: int) -> bytes:
  """Encode a signed integer so fixed-width BLOB ordering is numeric ordering."""
  # SQLite INTEGER 只有 64 位；偏移到无符号域后用 16 字节大端 BLOB 保存皮秒时间，
  # 可覆盖 int128，同时让 SQLite 的字节序排序等价于有符号数值排序。
  if isinstance(value, bool) or not isinstance(value, int):
    raise SQLiteCacheError("picosecond value must be an integer")
  if value < _I128_MIN or value > _I128_MAX:
    raise SQLiteCacheError("picosecond value exceeds the signed 128-bit cache range")
  return (value + 2**127).to_bytes(16, "big")


def decode_ordered_i128(value: bytes | bytearray | memoryview) -> int:
  """Decode an order-preserving signed 128-bit BLOB."""
  raw = bytes(value)
  if len(raw) != 16:
    raise SQLiteCacheError("ordered signed integer must contain exactly 16 bytes")
  return int.from_bytes(raw, "big") - 2**127


def _encode_u64(value: Any) -> bytes:
  if (
    isinstance(value, bool)
    or not isinstance(value, int)
    or not 0 <= value <= _UINT64_MAX
  ):
    raise SQLiteCacheError("uint64 cache value is outside [0, 2**64-1]")
  return value.to_bytes(8, "big")


def _decode_u64(value: bytes | bytearray | memoryview) -> int:
  raw = bytes(value)
  if len(raw) != 8:
    raise SQLiteCacheError("uint64 cache value must contain exactly eight bytes")
  return int.from_bytes(raw, "big")


def _stat_storage_values(stat: Any, value: TypedStat) -> tuple[Any, ...]:
  """Map one already-decoded XStat directly to its SQLite columns."""
  # oneof 各分支落到互斥列；double 保存原始 IEEE 位，bytes 额外保存摘要供回读校验。
  value_case = stat.WhichOneof("value")
  double_bits = nonfinite_tag = uint64_value = int64_value = None
  string_value = bytes_value = bytes_sha256 = None
  ref_id = ref_value = ref_resolved = None

  if value_case == "double_value":
    numeric = float(stat.double_value)
    double_bits = struct.pack(">d", numeric)
    if not math.isfinite(numeric):
      nonfinite_tag = (
        "NaN" if math.isnan(numeric) else "Infinity" if numeric > 0 else "-Infinity"
      )
  elif value_case == "uint64_value":
    uint64_value = _encode_u64(int(stat.uint64_value))
  elif value_case == "int64_value":
    int64_value = int(stat.int64_value)
  elif value_case == "str_value":
    string_value = stat.str_value
  elif value_case == "bytes_value":
    bytes_value = bytes(stat.bytes_value)
    bytes_sha256 = hashlib.sha256(bytes_value).hexdigest()
  elif value_case == "ref_value":
    ref_id = _encode_u64(int(stat.ref_value))
    ref_value = value.value
    ref_resolved = int(bool(value.ref_resolved))

  return (
    value.value_type,
    double_bits,
    nonfinite_tag,
    uint64_value,
    int64_value,
    string_value,
    bytes_value,
    bytes_sha256,
    ref_id,
    ref_value,
    ref_resolved,
  )


def _stat_from_row(row: sqlite3.Row, *, origin: str) -> TypedStat:
  value_type = row["value_type"]
  ref_id = None
  ref_resolved = None
  if value_type == "unset":
    value = None
  elif value_type == "double":
    raw = bytes(row["double_bits"])
    if len(raw) != 8:
      raise SQLiteCacheError("cached double does not contain eight bytes")
    numeric = struct.unpack(">d", raw)[0]
    tag = row["nonfinite_tag"]
    if tag is None:
      if not math.isfinite(numeric):
        raise SQLiteCacheError("cached finite double contains a non-finite payload")
      value = numeric
    else:
      expected = (
        "NaN" if math.isnan(numeric) else "Infinity" if numeric > 0 else "-Infinity"
      )
      if math.isfinite(numeric) or tag != expected:
        raise SQLiteCacheError("cached non-finite double tag disagrees with its bits")
      value = {"non_finite": tag}
  elif value_type == "uint64":
    value = _decode_u64(row["uint64_value"])
  elif value_type == "int64":
    value = row["int64_value"]
  elif value_type == "string":
    value = row["string_value"]
  elif value_type == "bytes":
    raw = bytes(row["bytes_value"])
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row["bytes_sha256"]:
      raise SQLiteCacheError("cached bytes stat checksum does not match its payload")
    value = {
      "size_bytes": len(raw),
      "sha256": digest,
      "hex_prefix": raw[:16].hex(),
    }
  elif value_type == "ref":
    value = row["ref_value"]
    ref_id = _decode_u64(row["ref_id"])
    ref_resolved = bool(row["ref_resolved"])
    if ref_resolved != (value is not None):
      raise SQLiteCacheError("cached ref resolution and value disagree")
  else:
    raise SQLiteCacheError(f"cached stat has unknown value_type {value_type!r}")

  return TypedStat(
    metadata_id=row["metadata_id"],
    name=row["stat_name"],
    description=row["stat_description"],
    value_type=value_type,
    value=value,
    origin=origin,
    ref_id=ref_id,
    ref_resolved=ref_resolved,
  )


def _event_timing_values(
  event: Any,
  *,
  line_timestamp_ns: int,
  location: str,
  diagnostics: list[dict[str, Any]],
) -> tuple[Any, ...]:
  """Derive one mutually exclusive SQLite timing record from raw XEvent fields."""
  # XEvent 的 data oneof 可能是 offset，也可能是聚合次数。只有合法 offset 事件
  # 才能进入时间线；aggregate/untimed 仍被保留，但不会参与区间计算。
  data_case = event.WhichOneof("data")
  duration = int(event.duration_ps)
  raw_offset = int(event.offset_ps) if data_case == "offset_ps" else None
  occurrences = int(event.num_occurrences) if data_case == "num_occurrences" else None
  timing_valid = False
  start_blob = end_blob = None

  if data_case == "offset_ps" and duration >= 0:
    start = line_timestamp_ns * 1000 + raw_offset
    kind = "span" if duration > 0 else "instant"
    timing_valid = True
    reason = None
    start_blob = encode_ordered_i128(start)
    end_blob = encode_ordered_i128(start + duration)
  elif data_case == "offset_ps":
    kind = "untimed"
    reason = "NEGATIVE_DURATION"
    diagnostics.append(
      diagnostic(
        "NEGATIVE_EVENT_DURATION",
        "negative event duration makes this record unavailable to timeline queries",
        location=location,
        duration_ps=duration,
      )
    )
  elif data_case == "num_occurrences":
    kind = "aggregate"
    reason = "AGGREGATE_RECORD"
    if occurrences < 0:
      diagnostics.append(
        diagnostic(
          "NEGATIVE_NUM_OCCURRENCES",
          "negative aggregate occurrence count is preserved as invalid producer data",
          location=location,
          num_occurrences=occurrences,
        )
      )
    if duration < 0:
      diagnostics.append(
        diagnostic(
          "NEGATIVE_AGGREGATE_DURATION",
          "negative source-reported aggregate duration is preserved but not aggregated",
          location=location,
          duration_ps=duration,
        )
      )
  else:
    kind = "untimed"
    reason = "DATA_ONEOF_UNSET"
    diagnostics.append(
      diagnostic(
        "EVENT_DATA_UNSET",
        "XEvent data oneof is unset; record is retained as untimed",
        location=location,
      )
    )

  return (
    data_case,
    raw_offset,
    duration,
    occurrences,
    kind,
    int(timing_valid),
    start_blob,
    end_blob,
    reason,
  )


def _flow_key(flow: dict[str, Any]) -> str:
  return _canonical_json(
    {
      "name": flow.get("name"),
      "value_type": flow.get("value_type"),
      "value": flow.get("value"),
    }
  )


_STAT_INSERT = """
INSERT INTO stat (
  owner_kind, plane_index, owner_plane_index, owner_event_metadata_id,
  owner_event_pk, ordinal, metadata_id, value_type, double_bits,
  nonfinite_tag, uint64_value, int64_value, string_value, bytes_value,
  bytes_sha256, ref_id, ref_value, ref_resolved
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _write_stats(
  connection: sqlite3.Connection,
  stats: Iterable[Any],
  *,
  plane: Any,
  plane_index: int,
  owner_kind: str,
  owner_id: int,
  location: str,
  diagnostics: list[dict[str, Any]],
  decoded: tuple[TypedStat, ...] | None = None,
) -> tuple[TypedStat, ...]:
  values = (
    tuple(
      _normalize_stats(
        stats,
        plane=plane,
        origin=owner_kind,
        location=location,
        diagnostics=diagnostics,
      )
    )
    if decoded is None
    else decoded
  )
  owner_plane = owner_id if owner_kind == "plane" else None
  owner_metadata = owner_id if owner_kind == "event_metadata" else None
  owner_event = owner_id if owner_kind == "event" else None
  for ordinal, (stat, value) in enumerate(zip(stats, values, strict=True)):
    connection.execute(
      """
      INSERT INTO stat_definition(plane_index, metadata_id, name, description)
      VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING
      """,
      (plane_index, value.metadata_id, value.name, value.description),
    )
    connection.execute(
      _STAT_INSERT,
      (
        owner_kind,
        plane_index,
        owner_plane,
        owner_metadata,
        owner_event,
        ordinal,
        value.metadata_id,
        *_stat_storage_values(stat, value),
      ),
    )
  return values


def _insert_diagnostics(
  connection: sqlite3.Connection,
  diagnostics: Iterable[dict[str, Any]],
  *,
  next_ordinal: int,
  event_pk: int | None = None,
) -> int:
  written = 0
  for event_ordinal, item in enumerate(diagnostics):
    written += 1
    ordinal = next_ordinal + event_ordinal
    details = {
      key: value for key, value in item.items() if key not in {"code", "message"}
    }
    connection.execute(
      "INSERT INTO diagnostic(ordinal, code, message, details_json) VALUES (?, ?, ?, ?)",
      (ordinal, item["code"], item["message"], _canonical_json(details)),
    )
    if event_pk is not None:
      connection.execute(
        """
        INSERT INTO event_diagnostic(event_pk, ordinal, diagnostic_ordinal)
        VALUES (?, ?, ?)
        """,
        (event_pk, event_ordinal, ordinal),
      )
  return next_ordinal + written


def _create_database_file(path: Path) -> None:
  flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
  if hasattr(os, "O_CLOEXEC"):
    flags |= os.O_CLOEXEC
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  descriptor = os.open(path, flags, 0o600)
  try:
    os.fchmod(descriptor, 0o600)
  finally:
    os.close(descriptor)


def _write_xspace_database(
  path: Path,
  space: Any,
  *,
  profile_sha256: str,
  raw_size: int,
  unknown_fields_present: bool,
) -> None:
  """Traverse one decoded XSpace and write its persistent data directly."""
  # 整个建库过程位于单事务中：先写规范化实体，再建索引并做 FK/integrity 校验，
  # 任一步失败都会回滚并移除数据库及 journal/WAL sidecar。
  if sqlite3.sqlite_version_info < (3, 37, 0):
    raise SQLiteCacheError("SQLite 3.37 or newer is required for STRICT cache tables")
  if raw_size > 2**63 - 1:
    raise SQLiteCacheError("raw_size_bytes exceeds SQLite's signed integer range")
  path = Path(path)
  if not path.parent.is_dir():
    raise SQLiteCacheError("SQLite cache parent directory does not exist")

  _create_database_file(path)
  connection: sqlite3.Connection | None = None
  try:
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA application_id={SQLITE_APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version={SQLITE_SCHEMA_VERSION}")
    connection.execute("BEGIN IMMEDIATE")
    for sql in _TABLE_SQL:
      connection.execute(sql)

    for ordinal, hostname in enumerate(space.hostnames):
      connection.execute(
        "INSERT INTO hostname(ordinal, value) VALUES (?, ?)",
        (ordinal, hostname),
      )
    for kind, messages in (("error", space.errors), ("warning", space.warnings)):
      for ordinal, message in enumerate(messages):
        connection.execute(
          "INSERT INTO capture_message(kind, ordinal, value) VALUES (?, ?, ?)",
          (kind, ordinal, message),
        )

    next_diagnostic = 0
    if unknown_fields_present:
      next_diagnostic = _insert_diagnostics(
        connection,
        (
          diagnostic(
            "UNKNOWN_PROTO_FIELDS_PRESENT",
            "input contains fields unknown to the pinned parser schema; known fields were preserved",
            parser_schema_revision=PARSER_SCHEMA_REVISION,
          ),
        ),
        next_ordinal=next_diagnostic,
      )

    counts = {
      "planes": len(space.planes),
      "lines": 0,
      "events": 0,
      "stats": 0,
      "event_metadata": 0,
      "stat_metadata": 0,
    }
    kind_counts = {kind: 0 for kind in _KINDS}

    for plane_index, plane in enumerate(space.planes):
      plane_location = _location(plane_index)
      counts["lines"] += len(plane.lines)
      counts["event_metadata"] += len(plane.event_metadata)
      counts["stat_metadata"] += len(plane.stat_metadata)
      connection.execute(
        """
        INSERT INTO plane(
          plane_index, plane_id, name, line_count,
          event_metadata_count, stat_metadata_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
          plane_index,
          int(plane.id),
          plane.name,
          len(plane.lines),
          len(plane.event_metadata),
          len(plane.stat_metadata),
        ),
      )

      plane_diagnostics: list[dict[str, Any]] = []
      _write_stats(
        connection,
        plane.stats,
        plane=plane,
        plane_index=plane_index,
        owner_kind="plane",
        owner_id=plane_index,
        location=plane_location,
        diagnostics=plane_diagnostics,
      )
      counts["stats"] += len(plane.stats)

      for metadata_id in sorted(plane.event_metadata):
        metadata = plane.event_metadata[metadata_id]
        metadata_location = f"{plane_location}/event-metadata:{metadata_id}"
        metadata_bytes = bytes(metadata.metadata)
        connection.execute(
          """
          INSERT INTO event_metadata(
            plane_index, metadata_id, name, display_name,
            metadata_bytes, metadata_sha256
          ) VALUES (?, ?, ?, ?, ?, ?)
          """,
          (
            plane_index,
            metadata_id,
            metadata.name,
            metadata.display_name,
            metadata_bytes,
            hashlib.sha256(metadata_bytes).hexdigest(),
          ),
        )
        for child_ordinal, child_id in enumerate(metadata.child_id):
          connection.execute(
            """
            INSERT INTO event_metadata_child(
              plane_index, metadata_id, ordinal, child_id
            ) VALUES (?, ?, ?, ?)
            """,
            (plane_index, metadata_id, child_ordinal, int(child_id)),
          )
          if child_id not in plane.event_metadata:
            plane_diagnostics.append(
              diagnostic(
                "EVENT_METADATA_CHILD_NOT_FOUND",
                "XEventMetadata child_id is absent from its XPlane metadata map",
                location=metadata_location,
                child_id=child_id,
              )
            )
        _write_stats(
          connection,
          metadata.stats,
          plane=plane,
          plane_index=plane_index,
          owner_kind="event_metadata",
          owner_id=metadata_id,
          location=metadata_location,
          diagnostics=plane_diagnostics,
        )
        counts["stats"] += len(metadata.stats)

      seen_line_ids: set[int] = set()
      for line_index, line in enumerate(plane.lines):
        line_location = _location(plane_index, line_index)
        if line.id in seen_line_ids:
          plane_diagnostics.append(
            diagnostic(
              "LOGICAL_LINE_SEGMENT",
              "line id is repeated and will be treated as another segment of one logical line",
              location=line_location,
              line_id=line.id,
            )
          )
        seen_line_ids.add(line.id)
        if line.duration_ps < 0:
          plane_diagnostics.append(
            diagnostic(
              "NEGATIVE_LINE_DURATION",
              "negative XLine duration is preserved as a record-local warning",
              location=line_location,
              duration_ps=line.duration_ps,
            )
          )

        connection.execute(
          """
          INSERT INTO line(
            plane_index, line_index, line_id, display_id, name, display_name,
            timestamp_ns, duration_ps, event_count, span_count, instant_count,
            aggregate_count, untimed_count
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0)
          """,
          (
            plane_index,
            line_index,
            int(line.id),
            int(line.display_id),
            line.name,
            line.display_name,
            int(line.timestamp_ns),
            int(line.duration_ps),
            len(line.events),
          ),
        )
        line_kind_counts = {kind: 0 for kind in _KINDS}

        for event_ordinal, event in enumerate(line.events):
          event_location = _location(plane_index, line_index, event_ordinal)
          event_diagnostics: list[dict[str, Any]] = []
          metadata = plane.event_metadata.get(event.metadata_id)
          if metadata is None:
            metadata_stats: tuple[TypedStat, ...] = ()
            event_diagnostics.append(
              diagnostic(
                "EVENT_METADATA_NOT_FOUND",
                "XEvent metadata_id is absent from its XPlane metadata map",
                location=event_location,
                metadata_id=event.metadata_id,
              )
            )
          else:
            metadata_stats = tuple(
              _normalize_stats(
                metadata.stats,
                plane=plane,
                origin="event_metadata",
                location=(f"{plane_location}/event-metadata:{event.metadata_id}"),
                diagnostics=[],
              )
            )

          event_stats = tuple(
            _normalize_stats(
              event.stats,
              plane=plane,
              origin="event",
              location=event_location,
              diagnostics=event_diagnostics,
            )
          )
          timing = _event_timing_values(
            event,
            line_timestamp_ns=int(line.timestamp_ns),
            location=event_location,
            diagnostics=event_diagnostics,
          )
          cursor = connection.execute(
            """
            INSERT INTO event(
              plane_index, line_index, event_ordinal, metadata_id,
              data_case, raw_offset_ps, duration_ps, num_occurrences, kind,
              timing_valid, start_ps, end_ps, timing_unavailable_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
              plane_index,
              line_index,
              event_ordinal,
              int(event.metadata_id),
              *timing,
            ),
          )
          if cursor.lastrowid is None:
            raise SQLiteCacheError("SQLite did not assign an event primary key")
          event_pk = int(cursor.lastrowid)
          _write_stats(
            connection,
            event.stats,
            plane=plane,
            plane_index=plane_index,
            owner_kind="event",
            owner_id=event_pk,
            location=event_location,
            diagnostics=event_diagnostics,
            decoded=event_stats,
          )
          counts["stats"] += len(event.stats)

          hlo, flow, source_info = _enrich(
            (*metadata_stats, *event_stats),
            location=event_location,
            diagnostics=event_diagnostics,
          )
          if hlo is not None or source_info is not None:
            connection.execute(
              """
              INSERT INTO event_enrichment(
                event_pk, hlo_op, hlo_json, source_info_json
              ) VALUES (?, ?, ?, ?)
              """,
              (
                event_pk,
                None if hlo is None else hlo.get("op"),
                None if hlo is None else _canonical_json(hlo),
                None if source_info is None else _canonical_json(source_info),
              ),
            )
          for flow_ordinal, item in enumerate(flow):
            connection.execute(
              """
              INSERT INTO event_flow(event_pk, ordinal, flow_key, payload_json)
              VALUES (?, ?, ?, ?)
              """,
              (
                event_pk,
                flow_ordinal,
                _flow_key(item),
                _canonical_json(item),
              ),
            )

          next_diagnostic = _insert_diagnostics(
            connection,
            event_diagnostics,
            next_ordinal=next_diagnostic,
            event_pk=event_pk,
          )
          kind = timing[4]
          kind_counts[kind] += 1
          line_kind_counts[kind] += 1
          counts["events"] += 1

        connection.execute(
          """
          UPDATE line
          SET span_count = ?, instant_count = ?,
              aggregate_count = ?, untimed_count = ?
          WHERE plane_index = ? AND line_index = ?
          """,
          (
            line_kind_counts["span"],
            line_kind_counts["instant"],
            line_kind_counts["aggregate"],
            line_kind_counts["untimed"],
            plane_index,
            line_index,
          ),
        )

      next_diagnostic = _insert_diagnostics(
        connection,
        plane_diagnostics,
        next_ordinal=next_diagnostic,
      )

    validation = {
      "status": "valid",
      "check_version": CHECK_VERSION,
      "parser_message": PARSER_MESSAGE,
      "parser_schema_revision": PARSER_SCHEMA_REVISION,
      "parser_schema_sha256": PARSER_SCHEMA_SHA256,
      "checked_counts": counts,
      "record_local_warning_count": next_diagnostic,
      "validation_source": "checked",
    }
    connection.execute(
      """
      INSERT INTO profile(
        singleton, profile_sha256, raw_size_bytes, span_count, instant_count,
        aggregate_count, untimed_count, validation_json
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      """,
      (
        1,
        profile_sha256,
        raw_size,
        kind_counts["span"],
        kind_counts["instant"],
        kind_counts["aggregate"],
        kind_counts["untimed"],
        _canonical_json(validation),
      ),
    )
    for sql in _INDEX_SQL:
      connection.execute(sql)

    foreign_key_error = connection.execute("PRAGMA foreign_key_check").fetchone()
    if foreign_key_error is not None:
      raise SQLiteCacheError(
        f"created SQLite cache violates a foreign key: {foreign_key_error}"
      )
    connection.execute("COMMIT")
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
      raise SQLiteCacheError("created SQLite cache failed integrity_check")
    connection.close()
    connection = None

    sidecars = [Path(f"{path}{suffix}") for suffix in ("-journal", "-wal", "-shm")]
    if any(os.path.lexists(sidecar) for sidecar in sidecars):
      raise SQLiteCacheError("published SQLite cache must not retain journal sidecars")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
      os.fchmod(descriptor, 0o600)
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
  except Exception:
    if connection is not None:
      try:
        connection.execute("ROLLBACK")
      except sqlite3.Error:
        pass
      connection.close()
    for candidate in (
      path,
      Path(f"{path}-journal"),
      Path(f"{path}-wal"),
      Path(f"{path}-shm"),
    ):
      try:
        candidate.unlink()
      except FileNotFoundError:
        pass
    raise


def _load_canonical_json(raw: Any, *, name: str) -> Any:
  if not isinstance(raw, str):
    raise SQLiteCacheError(f"{name} must be stored as JSON text")
  value = _json_loads(raw)
  if _canonical_json(value) != raw:
    raise SQLiteCacheError(f"{name} is not in canonical JSON form")
  return value


_STAT_SELECT_COLUMNS = """
s.stat_pk, s.owner_kind, s.plane_index, s.owner_plane_index,
s.owner_event_metadata_id, s.owner_event_pk, s.ordinal, s.metadata_id,
s.value_type, s.double_bits, s.nonfinite_tag, s.uint64_value,
s.int64_value, s.string_value, s.bytes_value, s.bytes_sha256,
s.ref_id, s.ref_value, s.ref_resolved,
d.name AS stat_name, d.description AS stat_description
"""


EVENT_ROW_SELECT = """
  e.event_pk, e.plane_index, p.plane_id, p.name AS plane_name,
  e.line_index, l.line_id, l.name AS line_name,
  l.display_name AS line_display_name, l.timestamp_ns,
  e.event_ordinal, e.metadata_id,
  m.name AS metadata_name, m.display_name AS metadata_display_name,
  e.start_ps, e.end_ps, e.duration_ps, e.num_occurrences,
  e.data_case, e.raw_offset_ps, e.kind, e.timing_valid,
  e.timing_unavailable_reason
"""

EVENT_ROW_FROM = """
  FROM event AS e
  JOIN plane AS p ON p.plane_index = e.plane_index
  JOIN line AS l
    ON l.plane_index = e.plane_index AND l.line_index = e.line_index
  LEFT JOIN event_metadata AS m
    ON m.plane_index = e.plane_index AND m.metadata_id = e.metadata_id
  LEFT JOIN event_enrichment AS en ON en.event_pk = e.event_pk
"""


def iter_event_rows(
  connection: sqlite3.Connection,
  *,
  where_sql: str = "1",
  parameters: tuple[Any, ...] = (),
  order_sql: str = "e.plane_index, e.line_index, e.event_ordinal",
  limit: int | None = None,
  offset: int = 0,
) -> Iterator[sqlite3.Row]:
  """Stream SQLite event rows without constructing a profile-sized collection."""
  if limit is not None and limit < 0:
    raise SQLiteCacheError("event query limit must be nonnegative")
  if offset < 0:
    raise SQLiteCacheError("event query offset must be nonnegative")
  suffix = ""
  query_parameters: tuple[Any, ...] = parameters
  if limit is not None:
    suffix = " LIMIT ? OFFSET ?"
    query_parameters = (*parameters, limit, offset)
  elif offset:
    suffix = " LIMIT -1 OFFSET ?"
    query_parameters = (*parameters, offset)
  query = (
    f"SELECT {EVENT_ROW_SELECT} {EVENT_ROW_FROM} "
    f"WHERE {where_sql} ORDER BY {order_sql}{suffix}"
  )
  yield from connection.execute(query, query_parameters)


def count_event_rows(
  connection: sqlite3.Connection,
  *,
  where_sql: str = "1",
  parameters: tuple[Any, ...] = (),
) -> int:
  """Count event candidates using the same relational selector as row reads."""
  row = connection.execute(
    f"SELECT count(*) {EVENT_ROW_FROM} WHERE {where_sql}", parameters
  ).fetchone()
  assert row is not None
  return int(row[0])


def _event_stats(
  connection: sqlite3.Connection,
  *,
  event_pk: int,
  plane_index: int,
  metadata_id: int,
) -> tuple[TypedStat, ...]:
  query = f"""
    SELECT {_STAT_SELECT_COLUMNS}
    FROM stat AS s
    JOIN stat_definition AS d
      ON d.plane_index = s.plane_index AND d.metadata_id = s.metadata_id
    WHERE
      (s.owner_kind = 'event_metadata'
       AND s.plane_index = ? AND s.owner_event_metadata_id = ?)
      OR (s.owner_kind = 'event' AND s.owner_event_pk = ?)
    ORDER BY
      CASE s.owner_kind WHEN 'event_metadata' THEN 0 ELSE 1 END,
      s.ordinal
  """
  result: list[TypedStat] = []
  for row in connection.execute(query, (plane_index, metadata_id, event_pk)):
    result.append(_stat_from_row(row, origin=row["owner_kind"]))
  return tuple(result)


def hydrate_event_record(
  connection: sqlite3.Connection,
  row: sqlite3.Row,
  *,
  profile: ProfileRecord,
) -> EventRecord:
  """Hydrate one selected SQLite event row into the public domain structure."""
  event_pk = int(row["event_pk"])
  enrichment = connection.execute(
    "SELECT hlo_json, source_info_json FROM event_enrichment WHERE event_pk = ?",
    (event_pk,),
  ).fetchone()
  hlo = (
    None
    if enrichment is None or enrichment["hlo_json"] is None
    else _load_canonical_json(enrichment["hlo_json"], name="event_enrichment.hlo_json")
  )
  source_info = (
    None
    if enrichment is None or enrichment["source_info_json"] is None
    else _load_canonical_json(
      enrichment["source_info_json"], name="event_enrichment.source_info_json"
    )
  )
  flow = tuple(
    _load_canonical_json(item["payload_json"], name="event_flow.payload_json")
    for item in connection.execute(
      "SELECT payload_json FROM event_flow WHERE event_pk = ? ORDER BY ordinal",
      (event_pk,),
    )
  )
  diagnostics: list[dict[str, Any]] = []
  for item in connection.execute(
    """
    SELECT d.code, d.message, d.details_json
    FROM event_diagnostic AS ed
    JOIN diagnostic AS d ON d.ordinal = ed.diagnostic_ordinal
    WHERE ed.event_pk = ? ORDER BY ed.ordinal
    """,
    (event_pk,),
  ):
    details = _load_canonical_json(item["details_json"], name="diagnostic.details_json")
    if not isinstance(details, dict):
      raise SQLiteCacheError("diagnostic details must decode to an object")
    diagnostics.append({"code": item["code"], "message": item["message"], **details})
  metadata_name = row["metadata_name"]
  return EventRecord(
    profile_sha256=profile.profile_sha256,
    source_path=profile.source_path,
    plane_index=row["plane_index"],
    plane_id=row["plane_id"],
    plane_name=row["plane_name"],
    line_index=row["line_index"],
    line_id=row["line_id"],
    line_name=row["line_name"],
    line_display_name=row["line_display_name"],
    line_timestamp_ns=row["timestamp_ns"],
    event_ordinal=row["event_ordinal"],
    metadata_id=row["metadata_id"],
    name=(
      metadata_name
      if metadata_name is not None
      else f"<event-metadata:{row['metadata_id']}>"
    ),
    display_name=row["metadata_display_name"] or "",
    start_ps=(
      None if row["start_ps"] is None else decode_ordered_i128(row["start_ps"])
    ),
    end_ps=None if row["end_ps"] is None else decode_ordered_i128(row["end_ps"]),
    duration_ps=row["duration_ps"],
    num_occurrences=row["num_occurrences"],
    data_case=row["data_case"],
    raw_offset_ps=row["raw_offset_ps"],
    kind=row["kind"],
    timing_valid=bool(row["timing_valid"]),
    timing_unavailable_reason=row["timing_unavailable_reason"],
    stats=_event_stats(
      connection,
      event_pk=event_pk,
      plane_index=row["plane_index"],
      metadata_id=row["metadata_id"],
    ),
    hlo=hlo,
    flow=flow,
    source_info=source_info,
    diagnostics=tuple(diagnostics),
  )


_DATABASE_NAME = "profile.sqlite3"


def default_cache_root() -> Path:
  """Return the fixed per-user cache root for this tool."""
  # 固定路径是可移植性契约的一部分，不读取 XDG 或仓库相对配置。
  return Path.home() / ".cache" / "pallas-kernel" / "xprof-cli"


def _artifact_paths(cache_root: Path, profile_sha256: str) -> tuple[Path, Path]:
  profile_dir = cache_root / CACHE_SCHEMA_VERSION / profile_sha256[:2] / profile_sha256
  return profile_dir, profile_dir / _DATABASE_NAME


def _prepare_profile_directory(path: Path) -> None:
  try:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
  except OSError as error:
    raise CLIError(
      "CACHE_WRITE_FAILED",
      f"failed to prepare SQLite cache directory {path}: {error}",
    ) from error


def _open_source(path: Path) -> tuple[Path, int, str, int]:
  # 对同一个只读 fd 分块求哈希，并比较读取前后的 inode/size/mtime/ctime，
  # 从入口处拒绝分析过程中被改写的 profile。
  source_path = validate_input_path(path)
  flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  try:
    descriptor = os.open(source_path, flags)
    before = os.fstat(descriptor)
    if not stat_module.S_ISREG(before.st_mode):
      raise OSError("profile is not a regular file")
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, 1024 * 1024):
      digest.update(chunk)
      size += len(chunk)
    after = os.fstat(descriptor)
    if (
      before.st_dev,
      before.st_ino,
      before.st_size,
      before.st_mtime_ns,
      before.st_ctime_ns,
    ) != (
      after.st_dev,
      after.st_ino,
      after.st_size,
      after.st_mtime_ns,
      after.st_ctime_ns,
    ) or size != after.st_size:
      raise CLIError(
        "INPUT_CHANGED_DURING_READ",
        f"profile {source_path} changed while its content hash was being computed",
      )
    return source_path, descriptor, digest.hexdigest(), size
  except CLIError:
    if "descriptor" in locals():
      os.close(descriptor)
    raise
  except OSError as error:
    if "descriptor" in locals():
      os.close(descriptor)
    raise CLIError(
      "INPUT_READ_FAILED",
      f"failed to open and hash profile {source_path}: {error}",
    ) from error


def _decode_source_descriptor(
  descriptor: int,
  *,
  expected_sha256: str,
  raw_size: int,
  path: Path,
  xplane_module: Any,
) -> tuple[Any, bool]:
  """Verify and decode the source through a file-backed view, without a bytearray."""
  # mmap 避免复制整个大型 PB；映射后再次校验大小和摘要，封住 hash 与 decode 间隙。
  try:
    observed_size = os.fstat(descriptor).st_size
    if observed_size != raw_size:
      raise CLIError(
        "INPUT_CHANGED_DURING_READ",
        f"profile {path} changed size from {raw_size} to {observed_size} bytes "
        "between hashing and SQLite initialization",
      )
    if raw_size == 0:
      observed_sha256 = hashlib.sha256(b"").hexdigest()
      if observed_sha256 != expected_sha256:
        raise CLIError(
          "INPUT_CHANGED_DURING_READ",
          f"profile {path} changed between hashing and SQLite initialization",
        )
      return _decode_xspace(
        b"",
        source_path=path,
        profile_sha256=expected_sha256,
        xplane_module=xplane_module,
      )
    mapped = mmap.mmap(descriptor, raw_size, access=mmap.ACCESS_READ)
  except (OSError, ValueError) as error:
    raise CLIError(
      "INPUT_READ_FAILED",
      f"failed to map profile {path} for SQLite initialization: {error}",
    ) from error

  view = memoryview(mapped)
  try:
    observed_sha256 = hashlib.sha256(view).hexdigest()
    observed_size = os.fstat(descriptor).st_size
    if (
      len(view) != raw_size
      or observed_size != raw_size
      or (observed_sha256 != expected_sha256)
    ):
      raise CLIError(
        "INPUT_CHANGED_DURING_READ",
        f"profile {path} changed between hashing and SQLite initialization",
      )
    return _decode_xspace(
      view,
      source_path=path,
      profile_sha256=expected_sha256,
      xplane_module=xplane_module,
    )
  finally:
    view.release()
    mapped.close()


def _profile_reference(
  *,
  profile_sha256: str,
  source_path: Path,
  raw_size: int,
  database_path: Path,
  cache_status: str,
) -> ProfileRecord:
  return ProfileRecord(
    profile_sha256=profile_sha256,
    source_path=str(source_path),
    raw_size_bytes=raw_size,
    database_path=str(database_path),
    cache={
      "status": cache_status,
      "cache_schema_version": CACHE_SCHEMA_VERSION,
    },
  )


def _database_exists(path: Path) -> bool:
  try:
    metadata = path.lstat()
  except FileNotFoundError:
    return False
  except OSError as error:
    raise CLIError(
      "CACHE_READ_FAILED",
      f"failed to inspect SQLite cache {path}: {error}",
    ) from error
  if not stat_module.S_ISREG(metadata.st_mode):
    raise CLIError(
      "CACHE_READ_FAILED",
      f"SQLite cache artifact is not a regular file: {path}",
    )
  return True


def _initialize_database(
  database_path: Path,
  *,
  schema_dir: Path,
  source_descriptor: int,
  source_path: Path,
  profile_sha256: str,
  raw_size: int,
) -> None:
  # Resolve the pinned binding before mapping profile pages into the process.
  # 先解析并验证固定 schema，再把 PB 映射进进程，避免半初始化数据库被发布。
  xplane_module = load_xplane_binding(schema_dir)
  space, unknown_fields_present = _decode_source_descriptor(
    source_descriptor,
    expected_sha256=profile_sha256,
    raw_size=raw_size,
    path=source_path,
    xplane_module=xplane_module,
  )

  temporary = database_path.parent / (f".{database_path.name}.{uuid.uuid4().hex}.tmp")
  try:
    _write_xspace_database(
      temporary,
      space,
      profile_sha256=profile_sha256,
      raw_size=raw_size,
      unknown_fields_present=unknown_fields_present,
    )
    del space
    os.replace(temporary, database_path)
    directory_fd = os.open(
      database_path.parent,
      os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
      os.fsync(directory_fd)
    finally:
      os.close(directory_fd)
  except Exception:
    with contextlib.suppress(FileNotFoundError):
      temporary.unlink()
    raise


def load_profile(
  path: Path,
  *,
  cache_root: Path,
) -> ProfileRecord:
  """Open and hash a profile, creating its SQLite database only on a miss."""
  # 热路径只做输入哈希并复用现有数据库（打开时再验身份）；冷路径才解析并建索引。
  source_path, source_fd, profile_sha256, raw_size = _open_source(path)
  profile_dir, database_path = _artifact_paths(cache_root, profile_sha256)
  try:
    if _database_exists(database_path):
      return _profile_reference(
        profile_sha256=profile_sha256,
        source_path=source_path,
        raw_size=raw_size,
        database_path=database_path,
        cache_status="hit",
      )

    _prepare_profile_directory(profile_dir)
    try:
      _initialize_database(
        database_path,
        schema_dir=cache_root / CACHE_SCHEMA_VERSION / "schema",
        source_descriptor=source_fd,
        source_path=source_path,
        profile_sha256=profile_sha256,
        raw_size=raw_size,
      )
    except (OSError, sqlite3.Error) as error:
      raise CLIError(
        "CACHE_WRITE_FAILED",
        f"failed to initialize SQLite cache for {profile_sha256} "
        f"under {cache_root}: {error}",
      ) from error
    return _profile_reference(
      profile_sha256=profile_sha256,
      source_path=source_path,
      raw_size=raw_size,
      database_path=database_path,
      cache_status="miss",
    )
  finally:
    os.close(source_fd)


@contextlib.contextmanager
def open_profile_database(profile: ProfileRecord) -> Iterator[sqlite3.Connection]:
  """Open and identity-check the immutable SQLite artifact used by a query."""
  # 从已打开 fd 以 immutable/query_only 模式连接，避免路径替换和查询意外写缓存。
  descriptor: int | None = None
  try:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
      flags |= os.O_NOFOLLOW
    descriptor = os.open(Path(profile.database_path), flags)
    metadata = os.fstat(descriptor)
    if not stat_module.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
      raise SQLiteCacheError("SQLite cache is not a private regular file")

    descriptor_path = None
    for directory in ("/proc/self/fd", "/dev/fd"):
      candidate = f"{directory}/{descriptor}"
      if os.path.exists(candidate):
        descriptor_path = candidate
        break
    if descriptor_path is None:
      raise SQLiteCacheError(
        "this platform cannot open SQLite safely from a file descriptor"
      )

    connection = sqlite3.connect(
      f"file:{descriptor_path}?mode=ro&immutable=1",
      uri=True,
      isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
      connection.execute("PRAGMA query_only=ON")
      connection.execute("PRAGMA trusted_schema=OFF")
      connection.execute("PRAGMA foreign_keys=ON")
      connection.execute("PRAGMA temp_store=FILE")
      application_id = connection.execute("PRAGMA application_id").fetchone()[0]
      user_version = connection.execute("PRAGMA user_version").fetchone()[0]
      if (
        application_id != SQLITE_APPLICATION_ID or user_version != SQLITE_SCHEMA_VERSION
      ):
        raise SQLiteCacheError("SQLite application_id or user_version is incompatible")
      row = connection.execute(
        "SELECT profile_sha256, raw_size_bytes FROM profile WHERE singleton = 1"
      ).fetchone()
      if (
        row is None
        or row["profile_sha256"] != profile.profile_sha256
        or row["raw_size_bytes"] != profile.raw_size_bytes
      ):
        raise SQLiteCacheError("SQLite profile identity does not match the raw input")
      yield connection
    finally:
      connection.close()
  except (OSError, SQLiteCacheError, sqlite3.Error) as error:
    raise CLIError(
      "CACHE_READ_FAILED",
      f"failed to query SQLite cache {profile.database_path} "
      f"for {profile.profile_sha256}: {error}",
    ) from error
  finally:
    if descriptor is not None:
      os.close(descriptor)


def load_input_profile(path: Path) -> ProfileRecord:
  """Load the sole input and return its content-addressed SQLite reference."""
  requested_root = default_cache_root()
  try:
    root = requested_root.resolve()
  except (OSError, RuntimeError) as error:
    raise CLIError(
      "CACHE_WRITE_FAILED",
      f"CACHE_PATH_RESOLUTION_FAILED: cache root {requested_root} "
      f"could not be resolved: {error}",
    ) from error

  return load_profile(path, cache_root=root)


# ============================================================================
# Event queries
# ============================================================================

# 选择器尽量下推到唯一 profile 的 SQLite，只 hydrate 当前结果页。
# 分页 cursor 同时绑定 profile、筛选器和排序方式，禁止串用旧游标。

_MAX_JSON_SELECTOR_DEPTH = 64
_MAX_JSON_SELECTOR_NODES = 10_000
_MAX_CURSOR_LENGTH = 16 * 1024


@dataclass(frozen=True, slots=True)
class QueryPage:
  """One stable result page."""

  items: tuple[Any, ...]
  matched_count: int
  next_cursor: str | None
  truncated: bool
  warnings: tuple[dict[str, Any], ...]
  ordering: str


def _values(args: Any, name: str) -> list[Any]:
  value = getattr(args, name, None)
  if value is None:
    return []
  return list(value) if isinstance(value, (list, tuple)) else [value]


def selector_payload(args: Any) -> dict[str, Any]:
  """Return the public selector state used to bind pagination cursors."""
  # `limit` 故意不参与绑定，因此翻页时可以改变页大小而不改变结果集合身份。
  names = (
    "event_id",
    "plane",
    "plane_id",
    "plane_index",
    "line",
    "line_id",
    "line_index",
    "event",
    "metadata_id",
    "stat",
    "stat_value",
    "kind",
    "hlo_op",
  )
  payload = {name: _values(args, name) for name in names}
  payload.update(
    {
      "match": getattr(args, "match", "exact"),
      "start_ps": getattr(args, "start_ps", None),
      "end_ps": getattr(args, "end_ps", None),
      "time_relation": getattr(args, "time_relation", "overlap"),
      "min_duration_ps": getattr(args, "min_duration_ps", None),
      "max_duration_ps": getattr(args, "max_duration_ps", None),
    }
  )
  return payload


def _string_matcher(mode: str) -> Callable[[str, str], bool]:
  if mode == "exact":
    return lambda value, pattern: value == pattern
  if mode == "glob":
    return fnmatch.fnmatchcase
  if mode == "regex":
    compiled: dict[str, re.Pattern[str]] = {}

    def regex_match(value: str, pattern: str) -> bool:
      try:
        expression = compiled.setdefault(pattern, re.compile(pattern))
      except (re.error, RecursionError) as error:
        raise CLIError(
          "INVALID_ARGUMENT",
          f"INVALID_REGEX: invalid regular expression {pattern!r}: {error}",
        ) from error
      return expression.search(value) is not None

    return regex_match
  raise ValueError(f"unsupported string match mode: {mode!r}")


def _typed_cli_value(raw: str) -> Any:
  # --stat-value 使用严格 JSON：拒绝 NaN/Infinity，并限制嵌套深度与节点总数。
  def reject_nonstandard_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant {value!r}")

  try:
    value = json.loads(raw, parse_constant=reject_nonstandard_constant)
  except (json.JSONDecodeError, RecursionError, ValueError) as error:
    raise CLIError(
      "INVALID_ARGUMENT",
      f"INVALID_JSON_STAT_VALUE: --stat-value must be valid JSON: {raw!r}",
    ) from error
  finite, bounded = _json_value_properties(value)
  if not bounded:
    raise CLIError(
      "INVALID_ARGUMENT",
      "JSON_STAT_VALUE_TOO_COMPLEX: --stat-value exceeds the complexity bound",
    )
  if not finite:
    raise CLIError(
      "INVALID_ARGUMENT",
      "NON_FINITE_JSON_STAT_VALUE: --stat-value numbers must be finite",
    )
  return value


def _json_value_properties(value: Any) -> tuple[bool, bool]:
  """Return (all numbers finite, traversal within public complexity bounds)."""
  stack = [(value, 0)]
  visited = 0
  while stack:
    item, depth = stack.pop()
    visited += 1
    if visited > _MAX_JSON_SELECTOR_NODES or depth > _MAX_JSON_SELECTOR_DEPTH:
      return True, False
    if isinstance(item, float) and not math.isfinite(item):
      return False, True
    if isinstance(item, dict):
      stack.extend((child, depth + 1) for child in item.values())
    elif isinstance(item, (list, tuple)):
      stack.extend((child, depth + 1) for child in item)
  return True, True


def _numeric(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cursor_fingerprint(profile_sha256: str, binding: dict[str, Any]) -> str:
  raw = json.dumps(
    {"profile_sha256": profile_sha256, "binding": binding},
    sort_keys=True,
    separators=(",", ":"),
  ).encode()
  return hashlib.sha256(raw).hexdigest()


def _encode_cursor(offset: int, fingerprint: str) -> str:
  # cursor 是不透明状态而非安全凭证；校验和用于发现截断/误传，fingerprint 防串查询。
  payload = {"version": 1, "offset": offset, "fingerprint": fingerprint}
  payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  envelope = {
    "payload": base64.urlsafe_b64encode(payload_raw).decode().rstrip("="),
    "checksum": hashlib.sha256(payload_raw).hexdigest()[:16],
  }
  return (
    base64.urlsafe_b64encode(
      json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    )
    .decode()
    .rstrip("=")
  )


def _decode_cursor(cursor: str, fingerprint: str) -> int:
  try:
    if len(cursor) > _MAX_CURSOR_LENGTH:
      raise ValueError("cursor exceeds the supported size")
    padded = cursor + "=" * (-len(cursor) % 4)
    envelope = json.loads(base64.urlsafe_b64decode(padded))
    encoded_payload = envelope["payload"]
    payload_padded = encoded_payload + "=" * (-len(encoded_payload) % 4)
    payload_raw = base64.urlsafe_b64decode(payload_padded)
    if hashlib.sha256(payload_raw).hexdigest()[:16] != envelope["checksum"]:
      raise ValueError("checksum mismatch")
    payload = json.loads(payload_raw)
    if payload["version"] != 1 or payload["fingerprint"] != fingerprint:
      raise ValueError("cursor is bound to a different profile or query parameters")
    offset = payload["offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
      raise ValueError("invalid offset")
    return offset
  except (
    binascii.Error,
    KeyError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    RecursionError,
  ) as error:
    raise CLIError(
      "INVALID_CURSOR",
      f"cursor is malformed or incompatible with this query: {error}",
    ) from error


_LOCATOR = re.compile(r"\Aplane:(\d+)/line:(\d+)/event:(\d+)\Z")
_ZERO_TIME_KEY = encode_ordered_i128(0)


def _register_selector_functions(connection: Any, mode: str) -> None:
  # 把 glob/regex 和强类型 XStat 比较注册成确定性 SQLite 函数，供 WHERE 子句复用。
  matcher = _string_matcher(mode)

  def matches(value: Any, pattern: str) -> int:
    return int(matcher("" if value is None else str(value), pattern))

  def stat_value_matches(
    value_type: str,
    double_bits: Any,
    uint64_value: Any,
    int64_value: Any,
    string_value: Any,
    bytes_value: Any,
    ref_id: Any,
    ref_value: Any,
    expected_json: str,
  ) -> int:
    expected = json.loads(expected_json)
    if value_type == "double":
      value = struct.unpack(">d", bytes(double_bits))[0]
      return int(_numeric(expected) and math.isfinite(value) and value == expected)
    if value_type == "uint64":
      value = int.from_bytes(bytes(uint64_value), "big")
      return int(_numeric(expected) and value == expected)
    if value_type == "int64":
      return int(_numeric(expected) and int64_value == expected)
    if value_type == "string":
      return int(isinstance(expected, str) and string_value == expected)
    if value_type == "ref":
      if isinstance(expected, int) and not isinstance(expected, bool):
        return int(int.from_bytes(bytes(ref_id), "big") == expected)
      return int(isinstance(expected, str) and ref_value == expected)
    if value_type == "bytes":
      if not isinstance(expected, dict):
        return 0
      raw = bytes(bytes_value)
      value = {
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "hex_prefix": raw[:16].hex(),
      }
      return int(value == expected)
    if value_type == "unset":
      return int(expected is None)
    return 0

  connection.create_function("xprof_match", 2, matches, deterministic=True)
  connection.create_function(
    "xprof_stat_value_matches", 9, stat_value_matches, deterministic=True
  )


def _pattern_clause(
  expressions: tuple[str, ...],
  patterns: list[str],
  parameters: list[Any],
  *,
  mode: str,
) -> str | None:
  if not patterns:
    return None
  terms: list[str] = []
  for pattern in patterns:
    for expression in expressions:
      terms.append(
        f"{expression} = ?" if mode == "exact" else f"xprof_match({expression}, ?) = 1"
      )
      parameters.append(pattern)
  return "(" + " OR ".join(terms) + ")"


def _sqlite_selector(
  connection: Any,
  args: Any,
) -> tuple[str, tuple[Any, ...], list[dict[str, Any]]]:
  # 所有用户值通过参数绑定进入 SQL；这里只拼接固定列名、占位符和受控表达式。
  mode = getattr(args, "match", "exact")
  patterns = {
    name: [str(value) for value in _values(args, name)]
    for name in ("plane", "line", "event", "stat", "hlo_op")
  }
  if mode == "regex":
    for values in patterns.values():
      for pattern in values:
        try:
          re.compile(pattern)
        except (re.error, RecursionError) as error:
          raise CLIError(
            "INVALID_ARGUMENT",
            f"INVALID_REGEX: invalid regular expression {pattern!r}: {error}",
          ) from error
  clauses: list[str] = []
  parameters: list[Any] = []
  locator_values = _values(args, "event_id")
  if locator_values:
    locators: list[tuple[int, int, int]] = []
    for value in locator_values:
      match = _LOCATOR.fullmatch(str(value))
      if match is None:
        raise CLIError(
          "INVALID_ARGUMENT",
          f"INVALID_EVENT_ID: {value!r}; expected plane:<N>/line:<N>/event:<N>",
        )
      locator = tuple(int(match.group(index)) for index in (1, 2, 3))
      if any(index > _PARSER_INT64_MAX for index in locator):
        raise CLIError(
          "INVALID_ARGUMENT",
          f"INVALID_EVENT_ID: {value!r}; locator indices exceed int64",
        )
      locators.append(locator)
    clauses.append(
      "("
      + " OR ".join(
        "(e.plane_index = ? AND e.line_index = ? AND e.event_ordinal = ?)"
        for _ in locators
      )
      + ")"
    )
    for locator in locators:
      parameters.extend(locator)

  integer_columns = {
    "plane_id": "p.plane_id",
    "plane_index": "e.plane_index",
    "line_id": "l.line_id",
    "line_index": "e.line_index",
    "metadata_id": "e.metadata_id",
  }
  for name, column in integer_columns.items():
    values = _values(args, name)
    if values:
      clauses.append(f"{column} IN ({','.join('?' for _ in values)})")
      parameters.extend(values)
  kinds = _values(args, "kind")
  if kinds:
    clauses.append(f"e.kind IN ({','.join('?' for _ in kinds)})")
    parameters.extend(kinds)

  for expression, name in (
    (("p.name",), "plane"),
    (("l.name", "l.display_name"), "line"),
    (
      (
        "COALESCE(m.name, printf('<event-metadata:%d>', e.metadata_id))",
        "COALESCE(m.display_name, '')",
      ),
      "event",
    ),
    (("en.hlo_op",), "hlo_op"),
  ):
    clause = _pattern_clause(expression, patterns[name], parameters, mode=mode)
    if clause is not None:
      clauses.append(clause)

  minimum = getattr(args, "min_duration_ps", None)
  maximum = getattr(args, "max_duration_ps", None)
  if minimum is not None and maximum is not None and minimum > maximum:
    raise CLIError(
      "INVALID_ARGUMENT",
      "INVALID_DURATION_RANGE: --min-duration-ps must not exceed --max-duration-ps",
    )
  if minimum is not None:
    clauses.append("e.duration_ps >= ?")
    parameters.append(minimum)
  if maximum is not None:
    clauses.append("e.duration_ps <= ?")
    parameters.append(maximum)

  lower = getattr(args, "start_ps", None)
  upper = getattr(args, "end_ps", None)
  if lower is not None and upper is not None and lower > upper:
    raise CLIError(
      "INVALID_ARGUMENT",
      "INVALID_TIME_WINDOW: --start-ps must not exceed --end-ps",
    )
  if lower is not None or upper is not None:
    clauses.append("e.timing_valid = 1")
    if lower is not None and upper is not None and lower == upper:
      clauses.append("0")
    else:
      relation = getattr(args, "time_relation", "overlap")
      if relation == "starts-in":
        if lower is not None:
          clauses.append("e.start_ps >= ?")
          parameters.append(encode_ordered_i128(lower))
        if upper is not None:
          clauses.append("e.start_ps < ?")
          parameters.append(encode_ordered_i128(upper))
      elif relation == "contained":
        if lower is not None:
          clauses.append("e.start_ps >= ?")
          parameters.append(encode_ordered_i128(lower))
        if upper is not None:
          bound = encode_ordered_i128(upper)
          clauses.append(
            "((e.kind = 'instant' AND e.start_ps < ?) OR "
            "(e.kind != 'instant' AND e.end_ps <= ?))"
          )
          parameters.extend((bound, bound))
      elif relation == "overlap":
        if lower is not None:
          bound = encode_ordered_i128(lower)
          clauses.append(
            "((e.kind = 'instant' AND e.start_ps >= ?) OR "
            "(e.kind != 'instant' AND e.end_ps > ?))"
          )
          parameters.extend((bound, bound))
        if upper is not None:
          clauses.append("e.start_ps < ?")
          parameters.append(encode_ordered_i128(upper))
      else:
        raise ValueError(f"unknown time relation: {relation}")

  stat_patterns = patterns["stat"]
  stat_values = [_typed_cli_value(value) for value in _values(args, "stat_value")]
  if stat_patterns or stat_values:
    stat_clauses = [
      "((s.owner_kind = 'event' AND s.owner_event_pk = e.event_pk) OR "
      "(s.owner_kind = 'event_metadata' AND s.plane_index = e.plane_index "
      "AND s.owner_event_metadata_id = e.metadata_id))"
    ]
    stat_parameters: list[Any] = []
    name_clause = _pattern_clause(
      ("d.name",), stat_patterns, stat_parameters, mode=mode
    )
    if name_clause is not None:
      stat_clauses.append(name_clause)
    if stat_values:
      value_terms: list[str] = []
      for value in stat_values:
        value_terms.append(
          "xprof_stat_value_matches(s.value_type, s.double_bits, "
          "s.uint64_value, s.int64_value, s.string_value, s.bytes_value, "
          "s.ref_id, s.ref_value, ?) = 1"
        )
        stat_parameters.append(
          json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
      stat_clauses.append("(" + " OR ".join(value_terms) + ")")
    clauses.append(
      "EXISTS (SELECT 1 FROM stat AS s JOIN stat_definition AS d "
      "ON d.plane_index = s.plane_index AND d.metadata_id = s.metadata_id "
      "WHERE " + " AND ".join(stat_clauses) + ")"
    )
    parameters.extend(stat_parameters)

  warnings: list[dict[str, Any]] = []
  return " AND ".join(clauses) if clauses else "1", tuple(parameters), warnings


def _stat_type_compatible(value_type: str, expected: Any) -> bool:
  if value_type in {"double", "uint64", "int64"}:
    return _numeric(expected)
  if value_type == "string":
    return isinstance(expected, str)
  if value_type == "ref":
    return isinstance(expected, str) or (
      isinstance(expected, int) and not isinstance(expected, bool)
    )
  if value_type == "bytes":
    return isinstance(expected, dict)
  return value_type == "unset" and expected is None


def _selector_readiness(
  connection: Any,
  args: Any,
) -> tuple[bool, set[tuple[str, str]]]:
  hlo_available = (
    connection.execute(
      "SELECT 1 FROM event_enrichment WHERE hlo_op IS NOT NULL LIMIT 1"
    ).fetchone()
    is not None
  )
  if not _values(args, "stat_value"):
    return hlo_available, set()
  parameters: list[Any] = []
  patterns = [str(value) for value in _values(args, "stat")]
  clause = _pattern_clause(
    ("d.name",), patterns, parameters, mode=getattr(args, "match", "exact")
  )
  owner_clause = "s.owner_kind IN ('event_metadata', 'event')"
  where_sql = owner_clause if clause is None else f"{owner_clause} AND {clause}"
  observed = {
    (row["name"], row["value_type"])
    for row in connection.execute(
      """
      SELECT DISTINCT d.name, s.value_type
      FROM stat AS s
      JOIN stat_definition AS d
        ON d.plane_index = s.plane_index AND d.metadata_id = s.metadata_id
      WHERE """
      + where_sql,
      tuple(parameters),
    )
  }
  return hlo_available, observed


def _selector_warnings(
  args: Any,
  *,
  hlo_available: bool,
  observed_stat_types: set[tuple[str, str]],
) -> list[dict[str, Any]]:
  warnings: list[dict[str, Any]] = []
  if _values(args, "hlo_op") and not hlo_available:
    warnings.append(
      diagnostic(
        "HLO_ENRICHMENT_UNAVAILABLE",
        "--hlo-op was requested but no verifiable HLO XStat enrichment exists",
      )
    )
  expected_values = [_typed_cli_value(value) for value in _values(args, "stat_value")]
  if (
    expected_values
    and observed_stat_types
    and not any(
      _stat_type_compatible(value_type, expected)
      for _, value_type in observed_stat_types
      for expected in expected_values
    )
  ):
    warnings.append(
      diagnostic(
        "STAT_VALUE_TYPE_MISMATCH",
        "one or more --stat-value operands were incompatible with selected XStat oneof types",
        observed=[
          {"name": name, "value_type": value_type}
          for name, value_type in sorted(observed_stat_types)
        ],
        note="CLI values use JSON literals; quote a JSON string explicitly when needed",
      )
    )
  return warnings


def _sqlite_order(mode: str) -> str:
  stable = (
    "CASE e.kind WHEN 'span' THEN 0 WHEN 'instant' THEN 0 "
    "WHEN 'aggregate' THEN 1 ELSE 2 END, "
    f"COALESCE(e.start_ps, X'{_ZERO_TIME_KEY.hex()}'), "
    f"COALESCE(e.end_ps, X'{_ZERO_TIME_KEY.hex()}'), "
    "e.plane_index, e.line_index, e.event_ordinal"
  )
  if mode == "time":
    return stable
  if mode == "duration":
    return "e.duration_ps DESC, " + stable
  if mode == "name":
    return (
      "COALESCE(m.name, printf('<event-metadata:%d>', e.metadata_id)), "
      "COALESCE(m.display_name, ''), " + stable
    )
  raise ValueError(f"unsupported sort mode: {mode}")


def query_event_page(profile: ProfileRecord, args: Any) -> QueryPage:
  """Count, sort, and page events in SQLite; hydrate only the requested page."""
  # 选择、计数和分页都在唯一 profile 的 SQLite 中完成。
  limit = getattr(args, "limit", 100)
  if limit <= 0:
    raise ValueError("--limit must be positive")
  sort_mode = getattr(args, "sort", "time")
  binding = {
    "command": "events",
    "selectors": selector_payload(args),
    "sort": sort_mode,
  }
  fingerprint = _cursor_fingerprint(profile.profile_sha256, binding)
  offset = (
    _decode_cursor(getattr(args, "cursor", None), fingerprint)
    if getattr(args, "cursor", None)
    else 0
  )
  warnings: list[dict[str, Any]] = []
  with open_profile_database(profile) as connection:
    _register_selector_functions(connection, getattr(args, "match", "exact"))
    where_sql, parameters, profile_warnings = _sqlite_selector(connection, args)
    warnings.extend(profile_warnings)
    hlo_available, observed_stat_types = _selector_readiness(connection, args)
    warnings.extend(
      _selector_warnings(
        args,
        hlo_available=hlo_available,
        observed_stat_types=observed_stat_types,
      )
    )
    total = count_event_rows(connection, where_sql=where_sql, parameters=parameters)
    if offset > total:
      raise CLIError("INVALID_CURSOR", "cursor offset is beyond the current result set")
    page_rows = iter_event_rows(
      connection,
      where_sql=where_sql,
      parameters=parameters,
      order_sql=_sqlite_order(sort_mode),
      limit=limit,
      offset=offset,
    )
    items = tuple(
      hydrate_event_record(connection, row, profile=profile) for row in page_rows
    )
  end = offset + len(items)
  return QueryPage(
    items=items,
    matched_count=total,
    next_cursor=_encode_cursor(end, fingerprint) if end < total else None,
    truncated=end < total,
    warnings=tuple(warnings),
    ordering=(
      f"{sort_mode}; stable tie-breaker=(timing_bucket,start_ps,end_ps,"
      "plane_index,line_index,event_ordinal)"
    ),
  )


# ============================================================================
# Statistics queries
# ============================================================================

# stats 会完整扫描选中事件并在临时 SQLite 中分组。duration 是逐事件分布；
# interval/gap/concurrency/self-time 是区间派生量，只有时钟域可比较时才有意义。
# 精确模式保留整数与有理数语义；近似模式必须在输出中声明算法及采样上限。

_STAT_GROUP_FIELDS = frozenset({"stat", "stat-name", "stat-type", "stat-value"})
_GROUP_FIELDS = {
  "plane",
  "plane-id",
  "plane-index",
  "plane-name",
  "line",
  "line-id",
  "line-index",
  "line-name",
  "line-display-name",
  "event",
  "event-name",
  "event-display-name",
  "metadata-id",
  "stat",
  "stat-name",
  "stat-type",
  "stat-value",
  "kind",
  "hlo",
  "hlo-op",
  "hlo-module",
  "hlo-category",
}
_METRICS = {"duration", "interval", "gap", "concurrency", "self-time", "xstats"}
_PUBLIC_STAT_LIMIT = 100


@dataclass(frozen=True, slots=True)
class StatsResult:
  """A paged stats result plus bounded full-scan group state."""

  groups: tuple[dict[str, Any], ...]
  group_fields: tuple[str, ...]
  percentile_mode: str
  matched_count: int
  returned_count: int
  next_cursor: str | None
  truncated: bool
  warnings: tuple[dict[str, Any], ...]
  ordering: str
  scan_complete: bool
  scanned_event_count: int
  selected_event_count: int
  approximation: dict[str, Any]


def _parse_csv(
  raw: str | None, *, default: tuple[str, ...], allowed: set[str], option: str
) -> tuple[str, ...]:
  # 字段顺序属于分组键和 cursor 契约；拒绝未知、空值和重复项，避免隐式规范化。
  values = (
    default
    if raw is None
    else tuple(value.strip() for value in raw.split(",") if value.strip())
  )
  unknown = sorted(set(values) - allowed)
  if unknown:
    raise CLIError(
      "INVALID_ARGUMENT",
      f"{option} contains unsupported values: {', '.join(unknown)}; "
      f"allowed: {', '.join(sorted(allowed))}",
    )
  if not values:
    raise CLIError("INVALID_ARGUMENT", f"{option} must not be empty")
  if len(set(values)) != len(values):
    raise CLIError("INVALID_ARGUMENT", f"{option} must not repeat values")
  return values


def _json_number(value: Fraction) -> int | float:
  """Encode an exact rational as a JSON number without losing integral values."""
  if value.denominator == 1:
    return value.numerator
  return value.numerator / value.denominator


def _exact_rational(value: Fraction) -> dict[str, str]:
  """Return a JSON-stable rational whose integers survive binary64 decoders."""
  return {
    "numerator": str(value.numerator),
    "denominator": str(value.denominator),
  }


def _integer_percentile(
  sorted_values: list[int], quantile: Fraction
) -> Fraction | None:
  """Compute an exact R7 percentile for an integral sample."""
  # 插值过程使用 Fraction，直到 JSON 渲染前都不引入二进制浮点舍入。
  if not sorted_values:
    return None
  if len(sorted_values) == 1:
    return Fraction(sorted_values[0])
  position = (len(sorted_values) - 1) * quantile
  lower = position.numerator // position.denominator
  upper = -(-position.numerator // position.denominator)
  if lower == upper:
    return Fraction(sorted_values[lower])
  weight = position - lower
  return (
    Fraction(sorted_values[lower]) * (1 - weight)
    + Fraction(sorted_values[upper]) * weight
  )


def _percentile(
  sorted_values: list[int | float], quantile: Fraction
) -> float | int | None:
  if not sorted_values:
    return None
  if len(sorted_values) == 1:
    return sorted_values[0]
  position = (len(sorted_values) - 1) * quantile
  lower = position.numerator // position.denominator
  upper = -(-position.numerator // position.denominator)
  if lower == upper:
    return sorted_values[lower]
  weight = position - lower
  if all(
    isinstance(value, int) and not isinstance(value, bool) for value in sorted_values
  ):
    interpolated = _integer_percentile(sorted_values, quantile)
    assert interpolated is not None
    return _json_number(interpolated)
  float_weight = float(weight)
  return math.fsum(
    (
      sorted_values[lower] * (1.0 - float_weight),
      sorted_values[upper] * float_weight,
    )
  )


def _finish_approximation_summary(
  modes: collections.Counter[str],
  algorithms: set[str],
  *,
  requested_mode: str,
) -> dict[str, Any]:
  ordered_algorithms = sorted(algorithms)
  distribution_count = sum(modes.values())
  if not modes:
    actual_mode = "not-applicable"
  elif len(modes) == 1:
    actual_mode = next(iter(modes))
  else:
    actual_mode = "mixed"
  if not ordered_algorithms:
    algorithm: str | None = None
  elif len(ordered_algorithms) == 1:
    algorithm = ordered_algorithms[0]
  else:
    algorithm = "mixed"
  approximate_count = modes["approximate"]
  return {
    "requested_mode": requested_mode,
    "mode": actual_mode,
    "algorithm": algorithm,
    "algorithms": ordered_algorithms,
    "distribution_count": distribution_count,
    "mode_counts": dict(sorted(modes.items())),
    "exact_distribution_count": modes["exact"],
    "approximate_distribution_count": approximate_count,
    "maximum_percentile_samples_per_distribution": (
      10_000 if approximate_count else None
    ),
  }


def _approximation_summary(
  rows: Iterable[dict[str, Any]], *, requested_mode: str
) -> dict[str, Any]:
  """Summarize algorithms used, including distributions hidden by display caps."""

  # 汇总遍历的是完整指标树，而不是只看公开列表前 N 项，避免低报近似计算数量。

  modes: collections.Counter[str] = collections.Counter()
  algorithms: set[str] = set()

  def collect(value: Any) -> None:
    if isinstance(value, dict):
      complete_summary = value.get("distribution_approximation")
      if isinstance(complete_summary, dict) and isinstance(
        complete_summary.get("mode_counts"), dict
      ):
        for mode, count in complete_summary["mode_counts"].items():
          if isinstance(mode, str) and isinstance(count, int) and count >= 0:
            modes[mode] += count
        algorithms.update(
          algorithm
          for algorithm in complete_summary.get("algorithms", [])
          if isinstance(algorithm, str)
        )
      percentiles = value.get("percentiles")
      if (
        isinstance(percentiles, dict)
        and isinstance(percentiles.get("mode"), str)
        and isinstance(percentiles.get("algorithm"), str)
      ):
        modes[percentiles["mode"]] += 1
        algorithms.add(percentiles["algorithm"])
      for key, child in value.items():
        if key not in {"percentiles", "distribution_approximation"} and not (
          complete_summary is not None and key == "stats"
        ):
          collect(child)
    elif isinstance(value, (list, tuple)):
      for child in value:
        collect(child)

  for row in rows:
    collect(row.get("kinds"))
    collect(row.get("metrics"))

  return _finish_approximation_summary(modes, algorithms, requested_mode=requested_mode)


def _numeric_stat_value(value: Any) -> bool:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return False
  return True if isinstance(value, int) else math.isfinite(value)


def _group_variants(
  event: Any, fields: tuple[str, ...]
) -> list[tuple[tuple[str, ...], dict[str, Any], TypedStat | None]]:
  # 复合字段携带完整父作用域；细粒度 *-name/*-id 则允许跨父作用域主动聚合。
  # 若按 XStat 分组，一个事件会为每个 stat 生成独立 variant。
  stat_variants: list[TypedStat | None] = (
    list(event.stats)
    if not _STAT_GROUP_FIELDS.isdisjoint(fields) and event.stats
    else [None]
  )
  variants: list[tuple[tuple[str, ...], dict[str, Any], TypedStat | None]] = []
  for stat in stat_variants:
    profile_scope = {"profile_sha256": event.profile_sha256}
    profile_identity = [event.profile_sha256]
    plane_scope = {
      **profile_scope,
      "plane_index": event.plane_index,
      "plane_id": event.plane_id,
      "plane_name": event.plane_name,
    }
    line_scope = {
      **plane_scope,
      "line_id": event.line_id,
      "line_name": event.line_name,
      "line_display_name": event.line_display_name,
    }
    event_scope = {
      **line_scope,
      "metadata_id": event.metadata_id,
      "event_name": event.name,
      "event_display_name": event.display_name,
    }
    key: list[str] = []
    display: dict[str, Any] = {}
    for field in fields:
      if field == "plane":
        value: Any = plane_scope
        identity = [
          *profile_identity,
          event.plane_index,
          event.plane_id,
          event.plane_name,
        ]
      elif field == "plane-id":
        value = event.plane_id
        identity = value
      elif field == "plane-index":
        value = event.plane_index
        identity = value
      elif field == "plane-name":
        value = event.plane_name
        identity = value
      elif field == "line":
        value = line_scope
        identity = [
          *profile_identity,
          event.plane_index,
          event.plane_id,
          event.plane_name,
          event.line_id,
          event.line_name,
          event.line_display_name,
        ]
      elif field == "line-id":
        value = event.line_id
        identity = value
      elif field == "line-index":
        value = event.line_index
        identity = value
      elif field == "line-name":
        value = event.line_name
        identity = value
      elif field == "line-display-name":
        value = event.line_display_name
        identity = value
      elif field == "event":
        value = event_scope
        identity = [
          *profile_identity,
          event.plane_index,
          event.plane_id,
          event.plane_name,
          event.line_id,
          event.line_name,
          event.line_display_name,
          event.metadata_id,
          event.name,
          event.display_name,
        ]
      elif field == "event-name":
        value = event.name
        identity = value
      elif field == "event-display-name":
        value = event.display_name
        identity = value
      elif field == "metadata-id":
        value = event.metadata_id
        identity = value
      elif field == "kind":
        value = event.kind
        identity = value
      elif field == "hlo":
        value = {**profile_scope, "hlo": event.hlo}
        identity = [*profile_identity, event.hlo]
      elif field in {"hlo-op", "hlo-module", "hlo-category"}:
        hlo_key = field.removeprefix("hlo-")
        value = None if event.hlo is None else event.hlo.get(hlo_key)
        identity = value
      elif field == "stat":
        value = (
          None
          if stat is None
          else {
            **event_scope,
            "stat_metadata_id": stat.metadata_id,
            "name": stat.name,
            "description": stat.description,
            "value_type": stat.value_type,
            "value": stat.value,
            "ref_id": stat.ref_id,
            "ref_resolved": stat.ref_resolved,
            "origin": stat.origin,
          }
        )
        identity = (
          None
          if stat is None
          else [
            *profile_identity,
            event.plane_index,
            event.plane_id,
            event.plane_name,
            event.line_id,
            event.line_name,
            event.line_display_name,
            event.metadata_id,
            event.name,
            event.display_name,
            stat.metadata_id,
            stat.name,
            stat.description,
            stat.value_type,
            stat.value,
            stat.ref_id,
            stat.ref_resolved,
            stat.origin,
          ]
        )
      elif field == "stat-name":
        value = None if stat is None else stat.name
        identity = value
      elif field == "stat-type":
        value = None if stat is None else stat.value_type
        identity = value
      elif field == "stat-value":
        value = (
          None
          if stat is None
          else (
            {
              "ref_id": stat.ref_id,
              "value": stat.value,
              "ref_resolved": stat.ref_resolved,
            }
            if stat.value_type == "ref"
            else stat.value
          )
        )
        identity = value
      else:  # Defensive guard: _parse_csv rejects unknown public fields.
        raise RuntimeError(f"unsupported normalized group field: {field}")
      display[field] = _bounded_value(value)
      key.append(_canonical_json(identity))
    variants.append((tuple(key), display, stat))
  return variants


@dataclass(frozen=True, slots=True)
class _RawGroupEvent:
  """Only the event identity fields needed to derive a group key."""

  profile_sha256: str
  plane_index: int
  plane_id: int
  plane_name: str
  line_index: int
  line_id: int
  line_name: str
  line_display_name: str
  metadata_id: int
  name: str
  display_name: str
  kind: str
  hlo: dict[str, Any] | None
  stats: tuple[TypedStat, ...] = ()


@dataclass(slots=True)
class _StatsWorkspace:
  """Ephemeral on-disk SQLite state for one stats invocation."""

  connection: sqlite3.Connection
  key_columns: tuple[str, ...]
  order_sql: str


def _create_stats_workspace(group_fields: tuple[str, ...]) -> _StatsWorkspace:
  # An empty SQLite filename creates an automatically deleted temporary file.
  # Keeping temp_store=FILE and a small page cache prevents a coarse group from
  # turning either grouping state or sort state into profile-sized Python RAM.
  # 空 SQLite 文件名会创建自动删除的临时库。配合 FILE temp_store 和小页缓存，
  # 即使粗粒度分组也不会把分组、排序状态膨胀成 profile 级 Python 内存。
  connection = sqlite3.connect("")
  connection.row_factory = sqlite3.Row
  connection.execute("PRAGMA temp_store=FILE")
  connection.execute("PRAGMA cache_size=-4096")
  key_columns = tuple(f"key_{index}" for index in range(len(group_fields)))
  key_definitions = ",\n      ".join(
    f"{column} TEXT NOT NULL" for column in key_columns
  )
  unique_columns = ", ".join(key_columns)
  connection.executescript(
    f"""
    CREATE TABLE group_def (
      group_id INTEGER PRIMARY KEY,
      key_json TEXT NOT NULL UNIQUE,
      {key_definitions},
      display_json TEXT NOT NULL,
      UNIQUE ({unique_columns})
    );
    CREATE TABLE work_event (
      event_seq INTEGER PRIMARY KEY,
      profile_sha256 TEXT NOT NULL,
      plane_index INTEGER NOT NULL,
      line_index INTEGER NOT NULL,
      line_id INTEGER NOT NULL,
      event_ordinal INTEGER NOT NULL,
      timing_scope TEXT NOT NULL,
      kind TEXT NOT NULL,
      timing_valid INTEGER NOT NULL,
      start_ps BLOB,
      end_ps BLOB,
      duration_ps INTEGER NOT NULL,
      num_occurrences INTEGER,
      timing_unavailable_reason TEXT,
      UNIQUE (
        profile_sha256, plane_index, line_index, event_ordinal
      )
    );
    CREATE TABLE group_event (
      group_id INTEGER NOT NULL,
      event_seq INTEGER NOT NULL,
      PRIMARY KEY (group_id, event_seq),
      FOREIGN KEY (group_id) REFERENCES group_def(group_id),
      FOREIGN KEY (event_seq) REFERENCES work_event(event_seq)
    ) WITHOUT ROWID;
    CREATE TABLE event_stat (
      event_seq INTEGER NOT NULL,
      stat_seq INTEGER NOT NULL,
      name TEXT NOT NULL,
      value_type TEXT NOT NULL,
      origin TEXT NOT NULL,
      numeric_valid INTEGER NOT NULL,
      int64_value INTEGER,
      uint64_value BLOB,
      double_value REAL,
      bytes_size INTEGER,
      value_key TEXT,
      rendered_json TEXT,
      PRIMARY KEY (event_seq, stat_seq),
      FOREIGN KEY (event_seq) REFERENCES work_event(event_seq)
    ) WITHOUT ROWID;
    CREATE TABLE group_stat (
      group_id INTEGER NOT NULL,
      event_seq INTEGER NOT NULL,
      stat_seq INTEGER NOT NULL,
      PRIMARY KEY (group_id, event_seq, stat_seq),
      FOREIGN KEY (group_id) REFERENCES group_def(group_id),
      FOREIGN KEY (event_seq, stat_seq)
        REFERENCES event_stat(event_seq, stat_seq)
    ) WITHOUT ROWID;
    CREATE TABLE page_group (
      group_id INTEGER PRIMARY KEY
    );
    CREATE TABLE self_time (
      event_seq INTEGER PRIMARY KEY,
      value_ps INTEGER NOT NULL
    );
    CREATE TABLE self_line_status (
      profile_sha256 TEXT NOT NULL,
      plane_index INTEGER NOT NULL,
      line_id INTEGER NOT NULL,
      reason TEXT NOT NULL,
      PRIMARY KEY (profile_sha256, plane_index, line_id)
    ) WITHOUT ROWID;
    CREATE TABLE scratch_self (
      event_seq INTEGER PRIMARY KEY,
      value_ps INTEGER NOT NULL
    );
    CREATE TABLE scratch_sample (
      metric TEXT NOT NULL,
      ordinal INTEGER NOT NULL,
      value_blob BLOB NOT NULL,
      PRIMARY KEY (metric, ordinal)
    ) WITHOUT ROWID;
    CREATE TABLE scratch_endpoint (
      coordinate BLOB NOT NULL,
      delta INTEGER NOT NULL
    );
    CREATE INDEX idx_group_event_event
      ON group_event(event_seq, group_id);
    CREATE INDEX idx_group_stat_event
      ON group_stat(event_seq, stat_seq, group_id);
    CREATE INDEX idx_event_stat_category
      ON event_stat(name, value_type, event_seq, stat_seq);
    CREATE INDEX idx_work_event_line
      ON work_event(profile_sha256, plane_index, line_id, line_index, event_ordinal);
    CREATE INDEX idx_scratch_endpoint_coordinate
      ON scratch_endpoint(coordinate);
    """
  )
  return _StatsWorkspace(
    connection=connection,
    key_columns=key_columns,
    order_sql=", ".join(key_columns),
  )


def _ensure_group(
  workspace: _StatsWorkspace,
  key: tuple[str, ...],
  display: dict[str, Any],
) -> int:
  """Persist one canonical group and return its SQLite id."""
  # key_json 用无损身份判等，display_json 仅负责有界展示，两者不能混用。
  connection = workspace.connection
  key_json = _canonical_json(key)
  columns = ", ".join(("key_json", *workspace.key_columns, "display_json"))
  placeholders = ", ".join("?" for _ in range(len(workspace.key_columns) + 2))
  cursor = connection.execute(
    f"INSERT OR IGNORE INTO group_def ({columns}) VALUES ({placeholders})",
    (key_json, *key, _canonical_json(display)),
  )
  if cursor.rowcount:
    return int(cursor.lastrowid)
  row = connection.execute(
    "SELECT group_id FROM group_def WHERE key_json = ?", (key_json,)
  ).fetchone()
  assert row is not None
  return int(row["group_id"])


def _map_group_variant(
  workspace: _StatsWorkspace,
  *,
  event_seq: int,
  event: _RawGroupEvent,
  group_fields: tuple[str, ...],
  stat: TypedStat | None,
  stat_seq: int | None,
  keep_grouped_stats: bool,
) -> None:
  grouped_event = replace(event, stats=() if stat is None else (stat,))
  variants = _group_variants(grouped_event, group_fields)
  assert len(variants) == 1
  key, display, _ = variants[0]
  group_id = _ensure_group(workspace, key, display)
  workspace.connection.execute(
    "INSERT OR IGNORE INTO group_event(group_id, event_seq) VALUES (?, ?)",
    (group_id, event_seq),
  )
  if keep_grouped_stats and stat_seq is not None:
    workspace.connection.execute(
      """
      INSERT INTO group_stat(group_id, event_seq, stat_seq)
      VALUES (?, ?, ?)
      """,
      (group_id, event_seq, stat_seq),
    )


_EVENT_STAT_QUERY = f"""
  SELECT {_STAT_SELECT_COLUMNS}
  FROM stat AS s
  JOIN stat_definition AS d
    ON d.plane_index = s.plane_index AND d.metadata_id = s.metadata_id
  WHERE
    (s.owner_kind = 'event_metadata'
     AND s.plane_index = ? AND s.owner_event_metadata_id = ?)
    OR (s.owner_kind = 'event' AND s.owner_event_pk = ?)
  ORDER BY
    CASE s.owner_kind WHEN 'event_metadata' THEN 0 ELSE 1 END,
    s.ordinal
"""


def _iter_source_stats(
  connection: sqlite3.Connection,
  *,
  event_pk: int,
  plane_index: int,
  metadata_id: int,
) -> Iterator[TypedStat]:
  rows = connection.execute(_EVENT_STAT_QUERY, (plane_index, metadata_id, event_pk))
  for row in rows:
    yield _stat_from_row(row, origin=str(row["owner_kind"]))


def _store_work_stat(
  workspace: _StatsWorkspace,
  *,
  event_seq: int,
  stat_seq: int,
  stat: TypedStat,
) -> None:
  numeric_valid = int(_numeric_stat_value(stat.value))
  int64_value: int | None = None
  uint64_value: bytes | None = None
  double_value: float | None = None
  bytes_size: int | None = None
  value_key: str | None = None
  rendered_json: str | None = None
  if stat.value_type == "int64":
    int64_value = int(stat.value)
  elif stat.value_type == "uint64":
    uint64_value = int(stat.value).to_bytes(8, "big")
  elif stat.value_type == "double" and numeric_valid:
    double_value = float(stat.value)
  elif stat.value_type == "bytes":
    bytes_size = int(stat.value["size_bytes"])
  elif stat.value_type in {"string", "ref"}:
    raw_value = (
      stat.value
      if stat.value_type == "string"
      else {"ref_id": stat.ref_id, "value": stat.value}
    )
    value_key = _canonical_json(raw_value)
    rendered_json = _canonical_json(_bounded_value(raw_value))
  workspace.connection.execute(
    """
    INSERT INTO event_stat(
      event_seq, stat_seq, name, value_type, origin, numeric_valid,
      int64_value, uint64_value, double_value, bytes_size,
      value_key, rendered_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
      event_seq,
      stat_seq,
      stat.name,
      stat.value_type,
      stat.origin,
      numeric_valid,
      int64_value,
      uint64_value,
      double_value,
      bytes_size,
      value_key,
      rendered_json,
    ),
  )


def _raw_group_event(
  profile: Any,
  row: sqlite3.Row,
  *,
  hlo: dict[str, Any] | None,
) -> _RawGroupEvent:
  metadata_name = row["metadata_name"]
  return _RawGroupEvent(
    profile_sha256=profile.profile_sha256,
    plane_index=int(row["plane_index"]),
    plane_id=int(row["plane_id"]),
    plane_name=str(row["plane_name"]),
    line_index=int(row["line_index"]),
    line_id=int(row["line_id"]),
    line_name=str(row["line_name"]),
    line_display_name=str(row["line_display_name"]),
    metadata_id=int(row["metadata_id"]),
    name=(
      str(metadata_name)
      if metadata_name is not None
      else f"<event-metadata:{row['metadata_id']}>"
    ),
    display_name=str(row["metadata_display_name"] or ""),
    kind=str(row["kind"]),
    hlo=hlo,
  )


def _ingest_selected_rows(
  workspace: _StatsWorkspace,
  profile: ProfileRecord,
  args: Any,
  *,
  group_fields: tuple[str, ...],
  metrics: tuple[str, ...],
  scan_limit: int | None,
) -> tuple[int, int]:
  """Scan raw candidates, then copy selected rows into the work database."""
  # scan-limit 限制候选评估而不是输出行数；一旦显式设置，最终必须标记扫描不完整。
  connection = workspace.connection
  stat_grouping = not _STAT_GROUP_FIELDS.isdisjoint(group_fields)
  keep_xstats = "xstats" in metrics
  need_stats = stat_grouping or keep_xstats
  need_hlo = any(
    field in {"hlo", "hlo-op", "hlo-module", "hlo-category"} for field in group_fields
  )
  selected_count = 0
  connection.execute("BEGIN")
  try:
    with open_profile_database(profile) as source:
      _register_selector_functions(source, getattr(args, "match", "exact"))
      where_sql, parameters, _ = _sqlite_selector(source, args)
      if scan_limit is None:
        candidate_row = source.execute("SELECT count(*) FROM event").fetchone()
      else:
        candidate_row = source.execute(
          """
          SELECT count(*)
          FROM (
            SELECT event_pk
            FROM event
            ORDER BY plane_index, line_index, event_ordinal
            LIMIT ?
          )
          """,
          (scan_limit,),
        ).fetchone()
      assert candidate_row is not None
      scanned_count = int(candidate_row[0])
      if scan_limit is not None:
        where_sql = (
          "e.event_pk IN ("
          "SELECT candidate.event_pk FROM event AS candidate "
          "ORDER BY candidate.plane_index, candidate.line_index, "
          "candidate.event_ordinal LIMIT ?"
          f") AND ({where_sql})"
        )
        parameters = (scanned_count, *parameters)

      hostname_count = int(
        source.execute("SELECT count(*) FROM hostname").fetchone()[0]
      )
      rows = iter_event_rows(
        source,
        where_sql=where_sql,
        parameters=parameters,
      )
      try:
        for row in rows:
          selected_count += 1
          timing_scope = (
            "profile"
            if hostname_count <= 1
            else f"plane:{row['plane_index']}/logical-line:{row['line_id']}"
          )
          cursor = connection.execute(
            """
            INSERT INTO work_event(
              profile_sha256, plane_index, line_index, line_id,
              event_ordinal, timing_scope, kind, timing_valid,
              start_ps, end_ps, duration_ps, num_occurrences,
              timing_unavailable_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
              profile.profile_sha256,
              row["plane_index"],
              row["line_index"],
              row["line_id"],
              row["event_ordinal"],
              timing_scope,
              row["kind"],
              row["timing_valid"],
              row["start_ps"],
              row["end_ps"],
              row["duration_ps"],
              row["num_occurrences"],
              row["timing_unavailable_reason"],
            ),
          )
          event_seq = int(cursor.lastrowid)
          hlo: dict[str, Any] | None = None
          if need_hlo:
            enrichment = source.execute(
              "SELECT hlo_json FROM event_enrichment WHERE event_pk = ?",
              (row["event_pk"],),
            ).fetchone()
            if enrichment is not None and enrichment["hlo_json"] is not None:
              loaded_hlo = json.loads(enrichment["hlo_json"])
              if isinstance(loaded_hlo, dict):
                hlo = loaded_hlo
          event = _raw_group_event(profile, row, hlo=hlo)

          if not stat_grouping:
            _map_group_variant(
              workspace,
              event_seq=event_seq,
              event=event,
              group_fields=group_fields,
              stat=None,
              stat_seq=None,
              keep_grouped_stats=False,
            )
          if not need_stats:
            continue

          had_stat = False
          for stat_seq, stat in enumerate(
            _iter_source_stats(
              source,
              event_pk=int(row["event_pk"]),
              plane_index=int(row["plane_index"]),
              metadata_id=int(row["metadata_id"]),
            )
          ):
            had_stat = True
            if keep_xstats:
              _store_work_stat(
                workspace,
                event_seq=event_seq,
                stat_seq=stat_seq,
                stat=stat,
              )
            if stat_grouping:
              _map_group_variant(
                workspace,
                event_seq=event_seq,
                event=event,
                group_fields=group_fields,
                stat=stat,
                stat_seq=stat_seq if keep_xstats else None,
                keep_grouped_stats=keep_xstats,
              )
          if stat_grouping and not had_stat:
            _map_group_variant(
              workspace,
              event_seq=event_seq,
              event=event,
              group_fields=group_fields,
              stat=None,
              stat_seq=None,
              keep_grouped_stats=keep_xstats,
            )
      finally:
        rows.close()
    connection.commit()
  except BaseException:
    connection.rollback()
    raise
  return scanned_count, selected_count


def _distribution_approximation(count: int, percentile_mode: str) -> dict[str, Any]:
  if percentile_mode == "exact":
    return {
      "mode": "exact",
      "requested_mode": percentile_mode,
      "algorithm": "exact-r7",
      "sample_count": count,
      "rank_error_bound": 0,
    }
  if count <= 10_000:
    return {
      "mode": "exact",
      "requested_mode": percentile_mode,
      "algorithm": "exact-r7-full-population-fallback",
      "sample_count": count,
      "population_count": count,
      "rank_error_bound": 0,
    }
  return {
    "mode": "approximate",
    "requested_mode": percentile_mode,
    "algorithm": "deterministic-stride-sample",
    "sample_count": 10_000,
    "population_count": count,
    "rank_error_bound": math.ceil(count / 10_000) / count,
  }


def _stream_distribution(
  connection: sqlite3.Connection,
  *,
  values_sql: str,
  parameters: tuple[Any, ...],
  decoder: Callable[[Any], int | float],
  percentile_mode: str,
  unit: str | None = "picoseconds",
  integer_values: bool,
) -> dict[str, Any]:
  """Compute a distribution with SQLite sorts and fixed-size Python state."""

  # exact 会按数据库顺序读取全部样本；大样本 approximate 只保留固定大小的确定性样本。

  def values(*, ordered: bool = False) -> Iterator[int | float]:
    query = f"SELECT value FROM ({values_sql}) AS distribution_values"
    if ordered:
      query += " ORDER BY value"
    for row in connection.execute(query, parameters):
      yield decoder(row[0])

  count = 0
  minimum: int | float | None = None
  maximum: int | float | None = None
  total_integer = 0
  sum_squares = 0
  for value in values():
    count += 1
    minimum = value if minimum is None else min(minimum, value)
    maximum = value if maximum is None else max(maximum, value)
    if integer_values:
      integer = int(value)
      total_integer += integer
      sum_squares += integer * integer
  if count == 0:
    return {"available": False, "reason": "NO_SAMPLES", "count": 0}
  assert minimum is not None and maximum is not None
  suffix = {"picoseconds": "_ps", "bytes": "_bytes", None: ""}[unit]

  def numeric_overflow(detected_in: str) -> dict[str, Any]:
    result: dict[str, Any] = {
      "available": False,
      "reason": "NUMERIC_OVERFLOW",
      "count": count,
      f"min{suffix}": minimum,
      f"max{suffix}": maximum,
      "overflow_detected_in": detected_in,
      "unavailable_fields": [
        f"total{suffix}",
        f"mean{suffix}",
        f"stddev{suffix}",
        f"p50{suffix}",
        f"p90{suffix}",
        f"p95{suffix}",
        f"p99{suffix}",
      ],
    }
    if unit is not None:
      result["unit"] = unit
    return result

  if integer_values:
    total: int | float = total_integer
    exact_mean = Fraction(total_integer, count)
    mean: int | float = _json_number(exact_mean)
    exact_variance = (
      Fraction(sum_squares * count - total_integer * total_integer, count * (count - 1))
      if count > 1
      else Fraction(0)
    )
    try:
      variance = float(exact_variance)
    except OverflowError:
      return numeric_overflow(f"stddev{suffix}")
  else:
    try:
      total = math.fsum(values())
    except (OverflowError, ValueError):
      return numeric_overflow(f"total{suffix}")
    if not math.isfinite(total):
      return numeric_overflow(f"total{suffix}")
    mean = total / count
    if not math.isfinite(mean):
      return numeric_overflow(f"mean{suffix}")
    try:
      variance = (
        math.fsum((value - mean) ** 2 for value in values()) / (count - 1)
        if count > 1
        else 0.0
      )
    except (OverflowError, ValueError):
      return numeric_overflow(f"stddev{suffix}")
    if not math.isfinite(variance):
      return numeric_overflow(f"stddev{suffix}")

  approximation = _distribution_approximation(count, percentile_mode)
  quantiles = (
    ("p50", Fraction(1, 2)),
    ("p90", Fraction(9, 10)),
    ("p95", Fraction(19, 20)),
    ("p99", Fraction(99, 100)),
  )
  percentile_results: dict[str, int | float] = {}
  exact_results: dict[str, Fraction] = {}
  if approximation["mode"] == "approximate":
    sample_count = 10_000
    targets = [
      round(index * (count - 1) / (sample_count - 1)) for index in range(sample_count)
    ]
    sampled: list[int | float] = []
    target_index = 0
    for ordinal, value in enumerate(values(ordered=True)):
      while target_index < len(targets) and targets[target_index] == ordinal:
        sampled.append(value)
        target_index += 1
      if target_index == len(targets):
        break
    for name, quantile in quantiles:
      percentile = _percentile(sampled, quantile)
      assert percentile is not None
      percentile_results[name] = percentile
      if integer_values:
        exact = _integer_percentile(sampled, quantile)
        assert exact is not None
        exact_results[name] = exact
  else:
    rank_pairs: dict[str, tuple[int, int, Fraction]] = {}
    required: set[int] = set()
    for name, quantile in quantiles:
      position = (count - 1) * quantile
      lower = position.numerator // position.denominator
      upper = -(-position.numerator // position.denominator)
      rank_pairs[name] = (lower, upper, position - lower)
      required.update((lower, upper))
    rank_values: dict[int, int | float] = {}
    for ordinal, value in enumerate(values(ordered=True)):
      if ordinal in required:
        rank_values[ordinal] = value
      if len(rank_values) == len(required):
        break
    for name, (lower, upper, weight) in rank_pairs.items():
      lower_value = rank_values[lower]
      upper_value = rank_values[upper]
      if integer_values:
        exact = (
          Fraction(int(lower_value))
          if lower == upper
          else (
            Fraction(int(lower_value)) * (1 - weight)
            + Fraction(int(upper_value)) * weight
          )
        )
        exact_results[name] = exact
        percentile_results[name] = _json_number(exact)
      elif lower == upper:
        percentile_results[name] = lower_value
      else:
        float_weight = float(weight)
        percentile_results[name] = math.fsum(
          (
            float(lower_value) * (1.0 - float_weight),
            float(upper_value) * float_weight,
          )
        )

  stddev = math.sqrt(max(0.0, variance))
  if not math.isfinite(stddev):
    return numeric_overflow(f"stddev{suffix}")
  result: dict[str, Any] = {
    "available": True,
    "count": count,
    f"total{suffix}": total,
    f"mean{suffix}": mean,
    f"min{suffix}": minimum,
    f"max{suffix}": maximum,
    f"stddev{suffix}": stddev,
    f"p50{suffix}": percentile_results["p50"],
    f"p90{suffix}": percentile_results["p90"],
    f"p95{suffix}": percentile_results["p95"],
    f"p99{suffix}": percentile_results["p99"],
    "percentiles": approximation,
  }
  if integer_values:
    exact_values = {"mean": exact_mean, **exact_results}
    for name, exact_value in exact_values.items():
      if exact_value.denominator != 1:
        result[f"{name}{suffix}_exact"] = _exact_rational(exact_value)
  if unit is not None:
    result["unit"] = unit
  return result


def _event_integer_distribution(
  connection: sqlite3.Connection,
  *,
  group_id: int,
  expression: str,
  predicate: str,
  percentile_mode: str,
  unit: str | None = "picoseconds",
) -> dict[str, Any]:
  return _stream_distribution(
    connection,
    values_sql=f"""
      SELECT {expression} AS value
      FROM group_event AS ge
      JOIN work_event AS we ON we.event_seq = ge.event_seq
      WHERE ge.group_id = ? AND {predicate}
    """,
    parameters=(group_id,),
    decoder=int,
    percentile_mode=percentile_mode,
    unit=unit,
    integer_values=True,
  )


def _scratch_distribution(
  connection: sqlite3.Connection,
  *,
  metric: str,
  percentile_mode: str,
) -> dict[str, Any]:
  return _stream_distribution(
    connection,
    values_sql=("SELECT value_blob AS value FROM scratch_sample WHERE metric = ?"),
    parameters=(metric,),
    decoder=decode_ordered_i128,
    percentile_mode=percentile_mode,
    integer_values=True,
  )


def _kind_summary_sql(
  connection: sqlite3.Connection,
  *,
  group_id: int,
  percentile_mode: str,
) -> dict[str, Any]:
  counts = collections.Counter(
    {
      str(row["kind"]): int(row["record_count"])
      for row in connection.execute(
        """
        SELECT we.kind, count(*) AS record_count
        FROM group_event AS ge
        JOIN work_event AS we ON we.event_seq = ge.event_seq
        WHERE ge.group_id = ?
        GROUP BY we.kind
        """,
        (group_id,),
      )
    }
  )
  occurrences_sum = 0
  valid_occurrences = 0
  for row in connection.execute(
    """
    SELECT we.num_occurrences
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.kind = 'aggregate'
    """,
    (group_id,),
  ):
    value = row["num_occurrences"]
    if value is not None and int(value) >= 0:
      occurrences_sum += int(value)
      valid_occurrences += 1
  untimed_reasons = {
    str(row["reason"]): int(row["record_count"])
    for row in connection.execute(
      """
      SELECT COALESCE(we.timing_unavailable_reason, 'UNKNOWN') AS reason,
             count(*) AS record_count
      FROM group_event AS ge
      JOIN work_event AS we ON we.event_seq = ge.event_seq
      WHERE ge.group_id = ? AND we.kind = 'untimed'
      GROUP BY reason ORDER BY reason
      """,
      (group_id,),
    )
  }
  aggregate_duration = _event_integer_distribution(
    connection,
    group_id=group_id,
    expression="we.duration_ps",
    predicate="we.kind = 'aggregate' AND we.duration_ps >= 0",
    percentile_mode=percentile_mode,
  )
  total_count = sum(counts.values())
  return {
    "record_count": total_count,
    "timed": {
      "record_count": counts["span"] + counts["instant"],
      "span_count": counts["span"],
      "instant_count": counts["instant"],
    },
    "aggregate": {
      "record_count": counts["aggregate"],
      "num_occurrences_sum": occurrences_sum,
      "invalid_occurrence_count": counts["aggregate"] - valid_occurrences,
      "source_reported_duration": aggregate_duration,
      "invalid_duration_count": (
        counts["aggregate"] - int(aggregate_duration.get("count", 0))
      ),
      "duration_semantics": (
        "producer-reported aggregate value; never mixed with timeline duration"
      ),
    },
    "untimed": {
      "record_count": counts["untimed"],
      "reasons": untimed_reasons,
    },
    "excluded_from_timeline_metrics": counts["aggregate"] + counts["untimed"],
  }


def _timing_scope_unavailable_sql(
  connection: sqlite3.Connection, *, group_id: int
) -> dict[str, Any] | None:
  # 多-host XSpace 无法将事件归属到 host；跨 logical line 的时间统计不可用。
  row = connection.execute(
    """
    SELECT count(DISTINCT we.timing_scope) AS scope_count,
           count(*) AS timed_count
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.timing_valid = 1
    """,
    (group_id,),
  ).fetchone()
  assert row is not None
  scope_count = int(row["scope_count"])
  if scope_count <= 1:
    return None
  return {
    "available": False,
    "reason": "CLOCK_ALIGNMENT_UNKNOWN",
    "timing_scope_count": scope_count,
    "timing_valid_record_count": int(row["timed_count"]),
  }


def _analysis_window_sql(
  connection: sqlite3.Connection, *, group_id: int, args: Any
) -> tuple[int, int] | None:
  first = connection.execute(
    """
    SELECT we.start_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.timing_valid = 1
    ORDER BY we.start_ps LIMIT 1
    """,
    (group_id,),
  ).fetchone()
  last = connection.execute(
    """
    SELECT we.end_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.timing_valid = 1
    ORDER BY we.end_ps DESC LIMIT 1
    """,
    (group_id,),
  ).fetchone()
  if first is None or last is None:
    return None
  requested_start = getattr(args, "start_ps", None)
  requested_end = getattr(args, "end_ps", None)
  return (
    decode_ordered_i128(first["start_ps"])
    if requested_start is None
    else requested_start,
    decode_ordered_i128(last["end_ps"]) if requested_end is None else requested_end,
  )


def _interval_metrics_sql(
  connection: sqlite3.Connection, *, group_id: int, args: Any
) -> dict[str, Any]:
  # span 按半开区间裁剪到分析窗口后求并集；instant 不贡献 active duration。
  window = _analysis_window_sql(connection, group_id=group_id, args=args)
  if window is None:
    return {"available": False, "reason": "NO_TIMED_EVENTS"}
  window_start, window_end = window
  active = 0
  total_duration = 0
  span_count = 0
  merged_count = 0
  merged_start: int | None = None
  merged_end: int | None = None
  coordinate_min: int | None = None
  coordinate_max: int | None = None
  for row in connection.execute(
    """
    SELECT we.start_ps, we.end_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.kind = 'span' AND we.timing_valid = 1
    ORDER BY we.start_ps, we.end_ps, we.event_seq
    """,
    (group_id,),
  ):
    start = max(window_start, decode_ordered_i128(row["start_ps"]))
    end = min(window_end, decode_ordered_i128(row["end_ps"]))
    if end <= start:
      continue
    span_count += 1
    total_duration += end - start
    coordinate_min = start if coordinate_min is None else min(coordinate_min, start)
    coordinate_max = end if coordinate_max is None else max(coordinate_max, end)
    if merged_start is None:
      merged_start, merged_end = start, end
      continue
    assert merged_end is not None
    if start > merged_end:
      active += merged_end - merged_start
      merged_count += 1
      merged_start, merged_end = start, end
    else:
      merged_end = max(merged_end, end)
  if merged_start is not None:
    assert merged_end is not None
    active += merged_end - merged_start
    merged_count += 1

  instant_count = 0
  for row in connection.execute(
    """
    SELECT we.start_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.kind = 'instant' AND we.timing_valid = 1
    """,
    (group_id,),
  ):
    point = decode_ordered_i128(row["start_ps"])
    instant_count += 1
    coordinate_min = point if coordinate_min is None else min(coordinate_min, point)
    coordinate_max = point if coordinate_max is None else max(coordinate_max, point)
  if coordinate_min is None or coordinate_max is None:
    return {
      "available": False,
      "reason": "NO_TIMED_EVENTS_IN_SELECTED_WINDOW",
      "selected_window": {"start_ps": window_start, "end_ps": window_end},
    }
  window_size = max(0, window_end - window_start)
  return {
    "available": True,
    "interval_semantics": "half-open [start_ps,end_ps); instants contribute zero",
    "span_count": span_count,
    "instant_count": instant_count,
    "wall_span_ps": coordinate_max - coordinate_min,
    "selected_window": {"start_ps": window_start, "end_ps": window_end},
    "selected_window_ps": window_size,
    "durations_clipped_to_selected_window": True,
    "active_interval_union_ps": active,
    "coverage": active / window_size if window_size else 0.0,
    "overlap_factor": total_duration / active if active else None,
    "merged_interval_count": merged_count,
  }


def _gap_metrics_sql(
  connection: sqlite3.Connection,
  *,
  group_id: int,
  percentile_mode: str,
) -> dict[str, Any]:
  connection.execute(
    "DELETE FROM scratch_sample WHERE metric IN ('interarrival', 'idle')"
  )
  timed_count = 0
  instant_count = 0
  previous_start: int | None = None
  interarrival_ordinal = 0
  for row in connection.execute(
    """
    SELECT we.kind, we.start_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.timing_valid = 1
    ORDER BY we.start_ps, we.end_ps, we.event_seq
    """,
    (group_id,),
  ):
    start = decode_ordered_i128(row["start_ps"])
    timed_count += 1
    instant_count += int(row["kind"] == "instant")
    if previous_start is not None:
      connection.execute(
        """
        INSERT INTO scratch_sample(metric, ordinal, value_blob)
        VALUES ('interarrival', ?, ?)
        """,
        (interarrival_ordinal, encode_ordered_i128(start - previous_start)),
      )
      interarrival_ordinal += 1
    previous_start = start

  merged_start: int | None = None
  merged_end: int | None = None
  idle_ordinal = 0
  merged_count = 0
  for row in connection.execute(
    """
    SELECT we.start_ps, we.end_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.kind = 'span' AND we.timing_valid = 1
    ORDER BY we.start_ps, we.end_ps, we.event_seq
    """,
    (group_id,),
  ):
    start = decode_ordered_i128(row["start_ps"])
    end = decode_ordered_i128(row["end_ps"])
    if merged_start is None:
      merged_start, merged_end = start, end
      merged_count = 1
      continue
    assert merged_end is not None
    if start > merged_end:
      connection.execute(
        """
        INSERT INTO scratch_sample(metric, ordinal, value_blob)
        VALUES ('idle', ?, ?)
        """,
        (idle_ordinal, encode_ordered_i128(start - merged_end)),
      )
      idle_ordinal += 1
      merged_start, merged_end = start, end
      merged_count += 1
    else:
      merged_end = max(merged_end, end)
  if merged_count:
    idle_semantics = "between consecutive merged positive-duration intervals"
  else:
    connection.execute(
      """
      INSERT INTO scratch_sample(metric, ordinal, value_blob)
      SELECT 'idle', ordinal, value_blob
      FROM scratch_sample WHERE metric = 'interarrival'
      """
    )
    idle_semantics = "between consecutive points for instant-only groups"
  return {
    "point_semantics": "instant start points participate with zero duration",
    "timed_event_count": timed_count,
    "instant_count": instant_count,
    "idle_gap_semantics": idle_semantics,
    "interarrival": _scratch_distribution(
      connection,
      metric="interarrival",
      percentile_mode=percentile_mode,
    ),
    "idle_gap": _scratch_distribution(
      connection,
      metric="idle",
      percentile_mode=percentile_mode,
    ),
  }


def _concurrency_metrics_sql(
  connection: sqlite3.Connection, *, group_id: int, args: Any
) -> dict[str, Any]:
  # 端点扫描遵循半开区间语义：同一坐标先结束旧区间，再开始新区间。
  window = _analysis_window_sql(connection, group_id=group_id, args=args)
  if window is None:
    return {"available": False, "reason": "NO_TIMED_WINDOW"}
  window_start, window_end = window
  instant_count = int(
    connection.execute(
      """
      SELECT count(*)
      FROM group_event AS ge
      JOIN work_event AS we ON we.event_seq = ge.event_seq
      WHERE ge.group_id = ? AND we.kind = 'instant' AND we.timing_valid = 1
      """,
      (group_id,),
    ).fetchone()[0]
  )
  connection.execute("DELETE FROM scratch_endpoint")
  interval_count = 0
  for row in connection.execute(
    """
    SELECT we.start_ps, we.end_ps
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.kind = 'span' AND we.timing_valid = 1
    """,
    (group_id,),
  ):
    start = max(window_start, decode_ordered_i128(row["start_ps"]))
    end = min(window_end, decode_ordered_i128(row["end_ps"]))
    if end <= start:
      continue
    connection.executemany(
      "INSERT INTO scratch_endpoint(coordinate, delta) VALUES (?, ?)",
      (
        (encode_ordered_i128(start), 1),
        (encode_ordered_i128(end), -1),
      ),
    )
    interval_count += 1
  if interval_count == 0:
    if instant_count:
      return {
        "available": True,
        "endpoint_semantics": (
          "half-open; end is applied before the following interval"
        ),
        "point_semantics": "instants contribute zero",
        "interval_count": 0,
        "instant_count": instant_count,
        "average": 0.0,
        "maximum": 0,
        "concurrency_time_integral_ps": 0,
      }
    if window_end <= window_start:
      return {"available": False, "reason": "ZERO_LENGTH_WINDOW"}
    return {"available": False, "reason": "NO_TIMED_SPANS"}

  current = 0
  maximum = 0
  area = 0
  previous = window_start
  for row in connection.execute(
    """
    SELECT coordinate, sum(delta) AS net_delta
    FROM scratch_endpoint
    GROUP BY coordinate ORDER BY coordinate
    """
  ):
    timestamp = decode_ordered_i128(row["coordinate"])
    area += current * (timestamp - previous)
    current += int(row["net_delta"])
    maximum = max(maximum, current)
    previous = timestamp
  area += current * (window_end - previous)
  return {
    "available": True,
    "endpoint_semantics": "half-open; end is applied before the following interval",
    "point_semantics": "instants contribute zero",
    "interval_count": interval_count,
    "instant_count": instant_count,
    "average": area / (window_end - window_start),
    "maximum": maximum,
    "concurrency_time_integral_ps": area,
  }


@dataclass(slots=True)
class _SQLSelfTimeFrame:
  line_index: int
  event_ordinal: int
  end_ps: int
  duration_ps: int
  child_union_ps: int = 0
  last_child_end_ps: int | None = None


def _prepare_self_times_sql(workspace: _StatsWorkspace, profile: ProfileRecord) -> None:
  """Compute page-relevant self times without retaining target event ids."""
  # 只为当前返回页涉及的逻辑行构建嵌套栈；交叉区间或完全相同区间会让 self time
  # 失去唯一解释，此时按行报告 unavailable，而不是强行相减。
  connection = workspace.connection
  profile_sha256 = profile.profile_sha256
  line_rows = connection.execute(
    """
    SELECT DISTINCT we.plane_index, we.line_id
    FROM page_group AS pg
    JOIN group_event AS ge ON ge.group_id = pg.group_id
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE we.kind = 'span' AND we.timing_valid = 1
    ORDER BY we.plane_index, we.line_id
    """
  )
  with open_profile_database(profile) as source:
    for line_row in line_rows:
      plane_index = int(line_row["plane_index"])
      line_id = int(line_row["line_id"])
      connection.execute("DELETE FROM scratch_self")
      stack: list[_SQLSelfTimeFrame] = []
      previous_interval: tuple[int, int] | None = None
      failure: str | None = None

      def finish_top() -> None:
        frame = stack.pop()
        self_time = max(0, frame.duration_ps - frame.child_union_ps)
        connection.execute(
          """
          INSERT OR REPLACE INTO scratch_self(event_seq, value_ps)
          SELECT we.event_seq, ?
          FROM work_event AS we
          WHERE we.plane_index = ?
            AND we.line_index = ? AND we.event_ordinal = ?
            AND EXISTS (
              SELECT 1
              FROM group_event AS ge
              JOIN page_group AS pg ON pg.group_id = ge.group_id
              WHERE ge.event_seq = we.event_seq
            )
          """,
          (
            self_time,
            plane_index,
            frame.line_index,
            frame.event_ordinal,
          ),
        )

      rows = iter_event_rows(
        source,
        where_sql=(
          "e.plane_index = ? AND l.line_id = ? AND e.kind = 'span' "
          "AND e.timing_valid = 1"
        ),
        parameters=(plane_index, line_id),
        order_sql=("e.start_ps, e.end_ps DESC, e.line_index, e.event_ordinal"),
      )
      try:
        for row in rows:
          start_ps = decode_ordered_i128(row["start_ps"])
          end_ps = decode_ordered_i128(row["end_ps"])
          interval = (start_ps, end_ps)
          if interval == previous_interval:
            failure = "AMBIGUOUS_IDENTICAL_INTERVALS"
            break
          previous_interval = interval
          while stack and start_ps >= stack[-1].end_ps:
            finish_top()
          if stack:
            parent = stack[-1]
            if end_ps > parent.end_ps:
              failure = "CROSSING_INTERVALS"
              break
            if parent.last_child_end_ps is None or start_ps >= parent.last_child_end_ps:
              parent.child_union_ps += end_ps - start_ps
            elif end_ps > parent.last_child_end_ps:
              parent.child_union_ps += end_ps - parent.last_child_end_ps
            parent.last_child_end_ps = max(parent.last_child_end_ps or end_ps, end_ps)
          stack.append(
            _SQLSelfTimeFrame(
              line_index=int(row["line_index"]),
              event_ordinal=int(row["event_ordinal"]),
              end_ps=end_ps,
              duration_ps=int(row["duration_ps"]),
            )
          )
      finally:
        rows.close()
      if failure is not None:
        connection.execute(
          """
          INSERT OR REPLACE INTO self_line_status(
            profile_sha256, plane_index, line_id, reason
          ) VALUES (?, ?, ?, ?)
          """,
          (profile_sha256, plane_index, line_id, failure),
        )
        continue
      while stack:
        finish_top()
      connection.execute(
        """
        INSERT OR REPLACE INTO self_time(event_seq, value_ps)
        SELECT event_seq, value_ps FROM scratch_self
        """
      )


def _self_time_metrics_sql(
  connection: sqlite3.Connection,
  *,
  group_id: int,
  percentile_mode: str,
) -> dict[str, Any]:
  span_count = int(
    connection.execute(
      """
      SELECT count(*)
      FROM group_event AS ge
      JOIN work_event AS we ON we.event_seq = ge.event_seq
      WHERE ge.group_id = ? AND we.kind = 'span' AND we.timing_valid = 1
      """,
      (group_id,),
    ).fetchone()[0]
  )
  if span_count == 0:
    return {"available": False, "reason": "NO_TIMED_SPANS"}
  line_count = int(
    connection.execute(
      """
      SELECT count(*) FROM (
        SELECT we.profile_sha256, we.plane_index, we.line_id
        FROM group_event AS ge
        JOIN work_event AS we ON we.event_seq = ge.event_seq
        WHERE ge.group_id = ? AND we.kind = 'span' AND we.timing_valid = 1
        GROUP BY we.profile_sha256, we.plane_index, we.line_id
      )
      """,
      (group_id,),
    ).fetchone()[0]
  )
  if line_count != 1:
    return {
      "available": False,
      "reason": "GROUP_SPANS_MULTIPLE_LOGICAL_LINES",
      "logical_line_count": line_count,
    }
  line = connection.execute(
    """
    SELECT we.profile_sha256, we.plane_index, we.line_id
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ? AND we.kind = 'span' AND we.timing_valid = 1
    LIMIT 1
    """,
    (group_id,),
  ).fetchone()
  assert line is not None
  status = connection.execute(
    """
    SELECT reason FROM self_line_status
    WHERE profile_sha256 = ? AND plane_index = ? AND line_id = ?
    """,
    (line["profile_sha256"], line["plane_index"], line["line_id"]),
  ).fetchone()
  if status is not None:
    return {"available": False, "reason": str(status["reason"])}
  result = _stream_distribution(
    connection,
    values_sql="""
      SELECT st.value_ps AS value
      FROM group_event AS ge
      JOIN self_time AS st ON st.event_seq = ge.event_seq
      WHERE ge.group_id = ?
    """,
    parameters=(group_id,),
    decoder=int,
    percentile_mode=percentile_mode,
    integer_values=True,
  )
  result["semantics"] = (
    "duration minus union of direct nested child spans on the full logical line"
  )
  return result


def _xstat_relation(grouped_stats: bool) -> str:
  if grouped_stats:
    return """
      FROM group_stat AS selected
      JOIN event_stat AS es
        ON es.event_seq = selected.event_seq
       AND es.stat_seq = selected.stat_seq
      WHERE selected.group_id = ?
    """
  return """
    FROM group_event AS selected
    JOIN event_stat AS es ON es.event_seq = selected.event_seq
    WHERE selected.group_id = ?
  """


def _xstat_distribution_sql(
  connection: sqlite3.Connection,
  *,
  relation: str,
  group_id: int,
  name: str,
  value_type: str,
  percentile_mode: str,
) -> dict[str, Any]:
  if value_type == "double":
    expression = "es.double_value"
    decoder: Callable[[Any], int | float] = float
    integer_values = False
  elif value_type == "uint64":
    expression = "es.uint64_value"

    def decode_uint64(value: Any) -> int:
      return int.from_bytes(bytes(value), "big")

    decoder = decode_uint64
    integer_values = True
  elif value_type == "int64":
    expression = "es.int64_value"
    decoder = int
    integer_values = True
  elif value_type == "bytes":
    expression = "es.bytes_size"
    decoder = int
    integer_values = True
  else:
    raise AssertionError(f"unsupported distribution XStat type: {value_type}")
  validity = "AND es.numeric_valid = 1" if value_type == "double" else ""
  return _stream_distribution(
    connection,
    values_sql=f"""
      SELECT {expression} AS value
      {relation}
        AND es.name = ? AND es.value_type = ? {validity}
    """,
    parameters=(group_id, name, value_type),
    decoder=decoder,
    percentile_mode=percentile_mode,
    unit="bytes" if value_type == "bytes" else None,
    integer_values=integer_values,
  )


def _xstat_metrics_sql(
  connection: sqlite3.Connection,
  *,
  group_id: int,
  percentile_mode: str,
  grouped_stats: bool,
) -> dict[str, Any]:
  relation = _xstat_relation(grouped_stats)
  category_sql = f"""
    SELECT es.name, es.value_type, count(*) AS stat_value_count,
           sum(es.numeric_valid) AS numeric_valid_count
    {relation}
    GROUP BY es.name, es.value_type
    ORDER BY es.name, es.value_type
  """
  modes: collections.Counter[str] = collections.Counter()
  algorithms: set[str] = set()
  returned_categories: list[tuple[str, str, int, int]] = []
  stat_count = 0
  for row in connection.execute(category_sql, (group_id,)):
    name = str(row["name"])
    value_type = str(row["value_type"])
    total_count = int(row["stat_value_count"])
    valid_count = (
      total_count
      if value_type in {"uint64", "int64", "bytes"}
      else int(row["numeric_valid_count"] or 0)
    )
    stat_count += 1
    if len(returned_categories) < _PUBLIC_STAT_LIMIT:
      returned_categories.append((name, value_type, total_count, valid_count))
    if value_type in {"double", "uint64", "int64", "bytes"} and valid_count:
      approximation = _distribution_approximation(valid_count, percentile_mode)
      modes[str(approximation["mode"])] += 1
      algorithms.add(str(approximation["algorithm"]))

  result_rows: list[dict[str, Any]] = []
  for name, value_type, total_count, valid_count in returned_categories:
    encoded_name = name.encode("utf-8", errors="surrogatepass")
    row: dict[str, Any] = {
      "name": (
        name
        if len(name) <= _PUBLIC_TEXT_LIMIT
        else name[: _PUBLIC_TEXT_LIMIT - 3] + "..."
      ),
      "name_truncated": len(name) > _PUBLIC_TEXT_LIMIT,
      "value_type": value_type,
      "count": total_count,
      "origins": {
        str(origin["origin"]): int(origin["origin_count"])
        for origin in connection.execute(
          f"""
          SELECT es.origin, count(*) AS origin_count
          {relation}
            AND es.name = ? AND es.value_type = ?
          GROUP BY es.origin ORDER BY es.origin
          """,
          (group_id, name, value_type),
        )
      },
    }
    if row["name_truncated"]:
      row["name_size_bytes"] = len(encoded_name)
      row["name_sha256"] = hashlib.sha256(encoded_name).hexdigest()
    if value_type in {"double", "uint64", "int64"}:
      row["numeric"] = _xstat_distribution_sql(
        connection,
        relation=relation,
        group_id=group_id,
        name=name,
        value_type=value_type,
        percentile_mode=percentile_mode,
      )
      row["excluded_non_finite_or_invalid"] = total_count - valid_count
    elif value_type in {"string", "ref"}:
      distinct_count = int(
        connection.execute(
          f"""
          SELECT count(DISTINCT es.value_key)
          {relation}
            AND es.name = ? AND es.value_type = ?
          """,
          (group_id, name, value_type),
        ).fetchone()[0]
      )
      top_values = [
        {
          "value": json.loads(value["rendered_json"]),
          "count": int(value["value_count"]),
        }
        for value in connection.execute(
          f"""
          SELECT es.value_key, min(es.rendered_json) AS rendered_json,
                 count(*) AS value_count
          {relation}
            AND es.name = ? AND es.value_type = ?
          GROUP BY es.value_key
          ORDER BY value_count DESC, es.value_key
          LIMIT 10
          """,
          (group_id, name, value_type),
        )
      ]
      row["distinct_count"] = distinct_count
      row["top_values"] = top_values
      row["top_values_truncated"] = distinct_count > 10
    elif value_type == "bytes":
      row["size"] = _xstat_distribution_sql(
        connection,
        relation=relation,
        group_id=group_id,
        name=name,
        value_type=value_type,
        percentile_mode=percentile_mode,
      )
      row["payloads_expanded"] = False
    result_rows.append(row)
  return {
    "stat_count": stat_count,
    "returned_stat_count": len(result_rows),
    "stats_truncated": len(result_rows) < stat_count,
    "distribution_approximation": _finish_approximation_summary(
      modes, algorithms, requested_mode=percentile_mode
    ),
    "stats": result_rows,
  }


def _stats_group_row_sql(
  connection: sqlite3.Connection,
  *,
  group_id: int,
  display: dict[str, Any],
  args: Any,
  metrics: tuple[str, ...],
  percentile_mode: str,
  group_fields: tuple[str, ...],
  scan_complete: bool,
) -> dict[str, Any]:
  counts = connection.execute(
    """
    SELECT sum(we.timing_valid) AS timing_valid_count,
           count(*) - sum(we.timing_valid) AS timing_excluded_count
    FROM group_event AS ge
    JOIN work_event AS we ON we.event_seq = ge.event_seq
    WHERE ge.group_id = ?
    """,
    (group_id,),
  ).fetchone()
  assert counts is not None
  timing_unavailable = _timing_scope_unavailable_sql(connection, group_id=group_id)
  row: dict[str, Any] = {
    "group": display,
    "kinds": _kind_summary_sql(
      connection, group_id=group_id, percentile_mode=percentile_mode
    ),
    "metrics": {},
    "completeness": {
      "scan_complete": scan_complete,
      "timing_valid_record_count": int(counts["timing_valid_count"] or 0),
      "timing_excluded_record_count": int(counts["timing_excluded_count"] or 0),
    },
  }
  if "duration" in metrics:
    row["metrics"]["duration"] = _event_integer_distribution(
      connection,
      group_id=group_id,
      expression="we.duration_ps",
      predicate="we.timing_valid = 1",
      percentile_mode=percentile_mode,
    )
  if "interval" in metrics:
    row["metrics"]["interval"] = (
      dict(timing_unavailable)
      if timing_unavailable is not None
      else _interval_metrics_sql(connection, group_id=group_id, args=args)
    )
  if "gap" in metrics:
    row["metrics"]["gap"] = (
      {
        "interarrival": dict(timing_unavailable),
        "idle_gap": dict(timing_unavailable),
      }
      if timing_unavailable is not None
      else _gap_metrics_sql(
        connection,
        group_id=group_id,
        percentile_mode=percentile_mode,
      )
    )
  if "concurrency" in metrics:
    row["metrics"]["concurrency"] = (
      dict(timing_unavailable)
      if timing_unavailable is not None
      else _concurrency_metrics_sql(connection, group_id=group_id, args=args)
    )
  if "self-time" in metrics:
    row["metrics"]["self_time"] = (
      dict(timing_unavailable)
      if timing_unavailable is not None
      else (
        _self_time_metrics_sql(
          connection,
          group_id=group_id,
          percentile_mode=percentile_mode,
        )
        if scan_complete
        else {"available": False, "reason": "SCAN_INCOMPLETE"}
      )
    )
  if "xstats" in metrics:
    row["metrics"]["xstats"] = _xstat_metrics_sql(
      connection,
      group_id=group_id,
      percentile_mode=percentile_mode,
      grouped_stats=not _STAT_GROUP_FIELDS.isdisjoint(group_fields),
    )
  return row


def compute_stats(profile: ProfileRecord, args: Any) -> StatsResult:
  """Group and aggregate through SQLite, retaining only the returned page."""
  # 分组全量完成后才分页；--limit 只约束返回组数，不改变任何组内聚合。
  group_fields = _parse_csv(
    getattr(args, "group_by", None),
    default=("plane", "line", "event"),
    allowed=_GROUP_FIELDS,
    option="--group-by",
  )
  metrics = _parse_csv(
    getattr(args, "metrics", None),
    default=("duration", "interval", "gap", "concurrency", "self-time", "xstats"),
    allowed=_METRICS,
    option="--metrics",
  )
  percentile_mode = getattr(args, "percentile_mode", "exact")
  scan_limit = getattr(args, "scan_limit", None)
  if scan_limit is not None and scan_limit < 0:
    raise ValueError("--scan-limit must be nonnegative")
  limit = getattr(args, "limit", 100)
  if limit <= 0:
    raise ValueError("--limit must be positive")

  warnings: list[dict[str, Any]] = []
  scan_complete = scan_limit is None
  workspace = _create_stats_workspace(group_fields)
  connection = workspace.connection
  try:
    scanned_event_count, selected_event_count = _ingest_selected_rows(
      workspace,
      profile,
      args,
      group_fields=group_fields,
      metrics=metrics,
      scan_limit=scan_limit,
    )
    if scan_limit is not None:
      warnings.append(
        diagnostic(
          "SCAN_LIMIT_APPLIED",
          "statistics are incomplete because --scan-limit bounded candidate evaluation",
          scan_limit=scan_limit,
          scanned_event_count=scanned_event_count,
          selected_event_count=selected_event_count,
          scan_order="plane_index,line_index,event_ordinal",
        )
      )

    binding = {
      "command": "stats",
      "selectors": selector_payload(args),
      "group_by": group_fields,
      "metrics": metrics,
      "percentile_mode": percentile_mode,
      "scan_limit": scan_limit,
    }
    fingerprint = _cursor_fingerprint(profile.profile_sha256, binding)
    cursor = getattr(args, "cursor", None)
    offset = _decode_cursor(cursor, fingerprint) if cursor else 0
    matched_count = int(
      connection.execute("SELECT count(*) FROM group_def").fetchone()[0]
    )
    if offset > matched_count:
      raise CLIError("INVALID_CURSOR", "cursor offset is beyond the current result set")
    end = min(matched_count, offset + limit)
    next_cursor = _encode_cursor(end, fingerprint) if end < matched_count else None
    page_records = list(
      connection.execute(
        f"""
        SELECT group_id, key_json, display_json
        FROM group_def
        ORDER BY {workspace.order_sql}
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
      )
    )
    connection.executemany(
      "INSERT INTO page_group(group_id) VALUES (?)",
      ((int(record["group_id"]),) for record in page_records),
    )
    if "self-time" in metrics and scan_complete and page_records:
      _prepare_self_times_sql(workspace, profile)

    groups = tuple(
      _stats_group_row_sql(
        connection,
        group_id=int(record["group_id"]),
        display=json.loads(record["display_json"]),
        args=args,
        metrics=metrics,
        percentile_mode=percentile_mode,
        group_fields=group_fields,
        scan_complete=scan_complete,
      )
      for record in page_records
    )
    approximation = _approximation_summary(groups, requested_mode=percentile_mode)
    return StatsResult(
      groups=groups,
      group_fields=group_fields,
      percentile_mode=percentile_mode,
      matched_count=matched_count,
      returned_count=len(page_records),
      next_cursor=next_cursor,
      truncated=end < matched_count,
      warnings=tuple(warnings),
      ordering="lexicographic canonical group key",
      scan_complete=scan_complete,
      scanned_event_count=scanned_event_count,
      selected_event_count=selected_event_count,
      approximation=approximation,
    )
  finally:
    connection.close()


# ============================================================================
# Context queries
# ============================================================================

# context 先把目标解析为唯一稳定 locator，再分别计算：跨 line 的同时段关系、
# 同一逻辑 line 的前后/嵌套关系，以及可选的原始 flow XStat 关联。
# 所有时间关系都限制在已证明可比较的 clock domain 内。

_FLOW_EVIDENCE_LIMIT = 100


@dataclass(frozen=True, slots=True)
class ContextResult:
  """A bounded context report around one unique target event."""

  target: EventRecord
  vertical: dict[str, Any] | None
  horizontal: dict[str, Any] | None
  flow_related: dict[str, Any] | None
  warnings: tuple[dict[str, Any], ...]
  matched_count: int
  returned_count: int
  truncated: bool
  ordering: str


def _has_non_locator_selector(args: Any) -> bool:
  payload = selector_payload(args)
  for key, value in payload.items():
    if key in {"event_id", "match", "time_relation"}:
      continue
    if value not in (None, []):
      return True
  return False


def _resolve_target(
  profile: ProfileRecord, args: Any
) -> tuple[EventRecord, list[dict[str, Any]]]:
  # context 不能接受模糊目标：零个或多个候选都会显式失败并要求更窄的 selector。
  event_ids = list(getattr(args, "event_id", None) or [])
  if len(event_ids) > 1:
    raise CLIError("INVALID_ARGUMENT", "context accepts at most one --event-id locator")
  if event_ids and _has_non_locator_selector(args):
    raise CLIError(
      "INVALID_ARGUMENT",
      "TARGET_SELECTOR_CONFLICT: use either --event-id or selectors, not both",
    )
  if not event_ids and not _has_non_locator_selector(args):
    raise CLIError(
      "INVALID_ARGUMENT",
      "TARGET_SELECTOR_REQUIRED: context needs --event-id or a target selector",
    )
  query_args = Namespace(**vars(args))
  query_args.limit = 20
  query_args.cursor = None
  query_args.sort = "time"
  page = query_event_page(profile, query_args)
  warnings = list(page.warnings)
  if page.matched_count == 0:
    raise CLIError(
      "EVENT_NOT_FOUND",
      "no target event matches this profile and selector",
    )
  if page.matched_count > 1:
    candidates = ", ".join(sorted(event.event_id for event in page.items[:3]))
    raise CLIError(
      "AMBIGUOUS_TARGET",
      f"selector matched {page.matched_count} events; choose one --event-id "
      f"(examples: {candidates})",
    )
  target = page.items[0]
  if not target.is_timed:
    raise CLIError(
      "EVENT_NOT_TIMED",
      f"{target.event_id} is {target.kind}, reason "
      f"{target.timing_unavailable_reason}; context requires a timed event",
    )
  return target, warnings


@dataclass(frozen=True, slots=True)
class _RawEventRow:
  """One timing-valid SQLite row retained before final EventRecord hydration."""

  row: sqlite3.Row
  start_ps: int
  end_ps: int

  @classmethod
  def from_sqlite(cls, row: sqlite3.Row) -> _RawEventRow:
    start = row["start_ps"]
    end = row["end_ps"]
    if start is None or end is None:
      raise RuntimeError("timing-valid context row is missing its interval")
    return cls(
      row=row,
      start_ps=decode_ordered_i128(start),
      end_ps=decode_ordered_i128(end),
    )

  @property
  def duration_ps(self) -> int:
    return int(self.row["duration_ps"])

  @property
  def kind(self) -> str:
    return str(self.row["kind"])

  @property
  def plane_index(self) -> int:
    return int(self.row["plane_index"])

  @property
  def line_index(self) -> int:
    return int(self.row["line_index"])

  @property
  def event_ordinal(self) -> int:
    return int(self.row["event_ordinal"])


def _temporal_relation(
  target: EventRecord, candidate: EventRecord | _RawEventRow
) -> tuple[str, int, str | None] | None:
  # span 使用半开区间；instant 是零长度点，并通过 instant_role 保留方向语义。
  assert target.start_ps is not None and target.end_ps is not None
  candidate_start = candidate.start_ps
  candidate_end = candidate.end_ps
  if target.kind == "instant" and candidate.kind == "instant":
    return ("exact", 0, None) if target.start_ps == candidate_start else None
  if target.kind == "instant":
    if candidate_start <= target.start_ps < candidate_end:
      return "instant_inside", 0, "target"
    return None
  if candidate.kind == "instant":
    if target.start_ps <= candidate_start < target.end_ps:
      return "instant_inside", 0, "candidate"
    return None
  if not (target.start_ps < candidate_end and candidate_start < target.end_ps):
    return None
  overlap = min(target.end_ps, candidate_end) - max(target.start_ps, candidate_start)
  if target.start_ps == candidate_start and target.end_ps == candidate_end:
    return "exact", overlap, None
  if candidate_start <= target.start_ps and candidate_end >= target.end_ps:
    return "contains", overlap, None
  if target.start_ps <= candidate_start and target.end_ps >= candidate_end:
    return "contained_by", overlap, None
  if candidate_start < target.start_ps:
    return "overlaps_start", overlap, None
  return "overlaps_end", overlap, None


def _relationship(
  target: EventRecord, candidate: EventRecord | _RawEventRow
) -> dict[str, Any] | None:
  relation = _temporal_relation(target, candidate)
  if relation is None:
    return None
  relation_name, overlap_ps, instant_role = relation
  assert target.start_ps is not None and target.end_ps is not None
  result: dict[str, Any] = {
    "relation": relation_name,
    "overlap_ps": overlap_ps,
    "overlap_ratio_target": (
      overlap_ps / target.duration_ps
      if target.duration_ps > 0
      else (1.0 if relation_name == "exact" else None)
    ),
    "start_delta_ps": candidate.start_ps - target.start_ps,
    "end_delta_ps": candidate.end_ps - target.end_ps,
    "_event": candidate,
  }
  if instant_role is not None:
    result["instant_role"] = instant_role
  return result


def _bounded_insert(
  rows: list[dict[str, Any]],
  item: dict[str, Any],
  *,
  limit: int,
  key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> None:
  """Keep only the lexicographically smallest ``limit`` lightweight rows."""
  # 在扫描总数不受限的同时，只保留最终可能返回的前 N 个轻量候选。
  if limit == 0:
    return
  item_key = key(item)
  position = bisect.bisect_right([key(row) for row in rows], item_key)
  if position >= limit:
    return
  rows.insert(position, item)
  if len(rows) > limit:
    rows.pop()


def _public_relation(
  item: dict[str, Any],
  *,
  connection: sqlite3.Connection,
  profile: Any,
  hydrated: dict[int, EventRecord],
) -> dict[str, Any]:
  """Hydrate and serialize one already-selected context relation."""
  event = item.get("_event")
  if isinstance(event, _RawEventRow):
    event_pk = int(event.row["event_pk"])
    normalized = hydrated.get(event_pk)
    if normalized is None:
      normalized = hydrate_event_record(connection, event.row, profile=profile)
      hydrated[event_pk] = normalized
    event = normalized
  if not isinstance(event, EventRecord):
    raise RuntimeError("context relation lost its selected SQLite event row")
  return {
    **{key: value for key, value in item.items() if key != "_event"},
    "event": event.to_dict(),
  }


def _vertical_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
  event = item["_event"]
  return (
    -item["overlap_ps"],
    abs(item["start_delta_ps"]),
    event.plane_index,
    event.line_index,
    event.event_ordinal,
  )


def _vertical_context(
  profile: ProfileRecord, target: EventRecord, limit: int
) -> dict[str, Any]:
  # vertical 查找其他逻辑 line 上与目标相交的事件，并按相交时长优先返回。
  if limit < 0:
    raise ValueError("--overlap-limit must be nonnegative")
  assert target.start_ps is not None and target.end_ps is not None
  relations: list[dict[str, Any]] = []
  matched_count = 0
  lower = encode_ordered_i128(target.start_ps)
  upper = encode_ordered_i128(target.end_ps)
  point_upper = encode_ordered_i128(target.start_ps + 1)
  overlap_sql = (
    "e.timing_valid = 1 AND NOT "
    "(e.plane_index = ? AND e.line_index = ? AND e.event_ordinal = ?) AND "
    "NOT (e.plane_index = ? AND l.line_id = ?) AND "
    "((e.kind = 'instant' AND e.start_ps >= ? AND e.start_ps < ?) OR "
    "(e.kind != 'instant' AND e.start_ps < ? AND e.end_ps > ?))"
  )
  parameters = (
    target.plane_index,
    target.line_index,
    target.event_ordinal,
    target.plane_index,
    target.line_id,
    lower,
    upper if target.kind != "instant" else point_upper,
    upper if target.kind != "instant" else point_upper,
    lower,
  )
  with open_profile_database(profile) as connection:
    hostname_count = int(
      connection.execute("SELECT count(*) FROM hostname").fetchone()[0]
    )
    if hostname_count > 1:
      return {
        "available": False,
        "reason": "CLOCK_ALIGNMENT_UNKNOWN",
        "hostname_count": hostname_count,
        "matched_count": 0,
        "returned_count": 0,
        "truncated": False,
        "relations": [],
      }
    for row in iter_event_rows(
      connection,
      where_sql=overlap_sql,
      parameters=parameters,
      order_sql="e.start_ps, e.end_ps, e.plane_index, e.line_index, e.event_ordinal",
    ):
      relation = _relationship(target, _RawEventRow.from_sqlite(row))
      if relation is not None:
        matched_count += 1
        _bounded_insert(relations, relation, limit=limit, key=_vertical_sort_key)
    hydrated: dict[int, EventRecord] = {}
    public_relations = [
      _public_relation(
        relation,
        connection=connection,
        profile=profile,
        hydrated=hydrated,
      )
      for relation in relations
    ]
  return {
    "available": True,
    "scope": "same-profile",
    "matched_count": matched_count,
    "returned_count": len(public_relations),
    "truncated": matched_count > len(public_relations),
    "ordering": "overlap_ps desc, abs(start_delta_ps), stable locator fields",
    "relations": public_relations,
  }


def _line_role(
  relation: str,
  instant_role: str | None,
) -> str:
  if relation == "contains":
    return "parent"
  if relation == "contained_by":
    return "child"
  if relation == "instant_inside":
    if instant_role == "target":
      return "parent"
    if instant_role == "candidate":
      return "child"
  return "sibling"


def _line_hierarchy_unavailable(
  connection: sqlite3.Connection, target: EventRecord
) -> str | None:
  """Return the first nesting violation with a disk-backed streaming stack."""
  # 同一逻辑 line 只有严格嵌套时才能解释 parent/child；交叉或同区间均视为歧义。
  stack = sqlite3.connect("")
  try:
    stack.execute("PRAGMA temp_store=FILE")
    stack.execute("PRAGMA cache_size=-64")
    stack.execute("PRAGMA journal_mode=OFF")
    stack.execute(
      """
      CREATE TABLE hierarchy_stack (
        depth INTEGER PRIMARY KEY CHECK (depth > 0),
        end_ps BLOB NOT NULL CHECK (length(end_ps) = 16)
      ) STRICT
      """
    )
    rows = connection.execute(
      """
      SELECT e.start_ps, e.end_ps
      FROM event AS e
      JOIN line AS l
        ON l.plane_index = e.plane_index AND l.line_index = e.line_index
      WHERE e.plane_index = ? AND l.line_id = ?
        AND e.kind = 'span' AND e.timing_valid = 1
      ORDER BY e.start_ps, e.end_ps DESC, e.line_index, e.event_ordinal
      """,
      (target.plane_index, target.line_id),
    )
    try:
      depth = 0
      top_end: int | None = None
      previous_interval: tuple[int, int] | None = None
      for row in rows:
        raw_start = row["start_ps"]
        raw_end = row["end_ps"]
        if raw_start is None or raw_end is None:
          raise RuntimeError("timing-valid context span is missing its interval")
        start = decode_ordered_i128(raw_start)
        end = decode_ordered_i128(raw_end)
        interval = (start, end)
        if interval == previous_interval:
          return "AMBIGUOUS_IDENTICAL_INTERVALS"
        previous_interval = interval

        while depth and top_end is not None and start >= top_end:
          depth -= 1
          if depth == 0:
            top_end = None
            break
          parent = stack.execute(
            "SELECT end_ps FROM hierarchy_stack WHERE depth = ?", (depth,)
          ).fetchone()
          if parent is None:
            raise RuntimeError("context hierarchy stack lost its parent interval")
          top_end = decode_ordered_i128(parent[0])

        if top_end is not None and end > top_end:
          return "CROSSING_INTERVALS"

        depth += 1
        stack.execute(
          "INSERT OR REPLACE INTO hierarchy_stack(depth, end_ps) VALUES (?, ?)",
          (depth, bytes(raw_end)),
        )
        top_end = end
      return None
    finally:
      rows.close()
  finally:
    stack.close()


def _horizontal_context(
  profile: ProfileRecord,
  target: EventRecord,
  *,
  before_limit: int,
  after_limit: int,
  window_before_ps: int | None,
  window_after_ps: int | None,
  overlap_limit: int,
) -> dict[str, Any]:
  # horizontal 在同一 plane + logical line 内拆分前驱、重叠和后继，并另报最近项。
  if before_limit < 0 or after_limit < 0:
    raise ValueError("--before and --after must be nonnegative")
  if window_before_ps is not None and window_before_ps < 0:
    raise ValueError("--window-before must be nonnegative")
  if window_after_ps is not None and window_after_ps < 0:
    raise ValueError("--window-after must be nonnegative")
  if overlap_limit < 0:
    raise ValueError("--overlap-limit must be nonnegative")
  assert target.start_ps is not None and target.end_ps is not None
  before: list[dict[str, Any]] = []
  after: list[dict[str, Any]] = []
  overlapping: list[dict[str, Any]] = []
  nearest_before: dict[str, Any] | None = None
  nearest_after: dict[str, Any] | None = None
  before_count = 0
  after_count = 0
  overlapping_count = 0

  def before_key(item: dict[str, Any]) -> tuple[Any, ...]:
    event = item["_event"]
    return (
      item["gap_ps"],
      -event.end_ps,
      -event.start_ps,
      event.line_index,
      event.event_ordinal,
    )

  def after_key(item: dict[str, Any]) -> tuple[Any, ...]:
    event = item["_event"]
    return (
      item["gap_ps"],
      event.start_ps,
      event.end_ps,
      event.line_index,
      event.event_ordinal,
    )

  def overlapping_key(item: dict[str, Any]) -> tuple[Any, ...]:
    event = item["_event"]
    return (
      event.start_ps,
      event.end_ps,
      event.line_index,
      event.event_ordinal,
    )

  with open_profile_database(profile) as connection:
    hierarchy_unavailable = _line_hierarchy_unavailable(connection, target)
    rows = iter_event_rows(
      connection,
      where_sql=(
        "e.plane_index = ? AND l.line_id = ? AND e.timing_valid = 1 AND NOT "
        "(e.plane_index = ? AND e.line_index = ? AND e.event_ordinal = ?)"
      ),
      parameters=(
        target.plane_index,
        target.line_id,
        target.plane_index,
        target.line_index,
        target.event_ordinal,
      ),
      order_sql="e.start_ps, e.end_ps, e.line_index, e.event_ordinal",
    )
    for row in rows:
      candidate = _RawEventRow.from_sqlite(row)
      relation = _relationship(target, candidate)
      if relation is not None:
        if hierarchy_unavailable is None:
          relation["line_role"] = _line_role(
            relation["relation"],
            relation.get("instant_role"),
          )
          relation["line_role_available"] = True
        else:
          relation["line_role"] = "unavailable"
          relation["line_role_available"] = False
          relation["line_role_reason"] = hierarchy_unavailable
        overlapping_count += 1
        _bounded_insert(
          overlapping,
          relation,
          limit=overlap_limit,
          key=overlapping_key,
        )
        continue
      if candidate.end_ps <= target.start_ps:
        gap = target.start_ps - candidate.end_ps
        if window_before_ps is None or gap <= window_before_ps:
          before_count += 1
          item = {"gap_ps": gap, "_event": candidate}
          if nearest_before is None or before_key(item) < before_key(nearest_before):
            nearest_before = item
          _bounded_insert(
            before,
            item,
            limit=before_limit,
            key=before_key,
          )
      elif candidate.start_ps >= target.end_ps:
        gap = candidate.start_ps - target.end_ps
        if window_after_ps is None or gap <= window_after_ps:
          after_count += 1
          item = {"gap_ps": gap, "_event": candidate}
          if nearest_after is None or after_key(item) < after_key(nearest_after):
            nearest_after = item
          _bounded_insert(
            after,
            item,
            limit=after_limit,
            key=after_key,
          )

    before.sort(
      key=lambda item: (
        item["_event"].start_ps,
        item["_event"].end_ps,
        item["_event"].line_index,
        item["_event"].event_ordinal,
      )
    )
    hydrated: dict[int, EventRecord] = {}

    def public(item: dict[str, Any]) -> dict[str, Any]:
      return _public_relation(
        item,
        connection=connection,
        profile=profile,
        hydrated=hydrated,
      )

    returned_before = [public(item) for item in before]
    nearest_before_pk = (
      int(nearest_before["_event"].row["event_pk"])
      if nearest_before is not None
      else None
    )
    nearest_predecessor = next(
      (
        row
        for item, row in zip(before, returned_before, strict=True)
        if int(item["_event"].row["event_pk"]) == nearest_before_pk
      ),
      None,
    )
    if nearest_predecessor is None and nearest_before is not None:
      nearest_predecessor = public(nearest_before)
    returned_after = [public(item) for item in after]
    nearest_after_pk = (
      int(nearest_after["_event"].row["event_pk"])
      if nearest_after is not None
      else None
    )
    nearest_successor = next(
      (
        row
        for item, row in zip(after, returned_after, strict=True)
        if int(item["_event"].row["event_pk"]) == nearest_after_pk
      ),
      None,
    )
    if nearest_successor is None and nearest_after is not None:
      nearest_successor = public(nearest_after)
    returned_overlapping = [public(item) for item in overlapping]

  return {
    "logical_line": {
      "profile_sha256": target.profile_sha256,
      "plane_index": target.plane_index,
      "line_id": target.line_id,
    },
    "before": {
      "matched_count": before_count,
      "returned_count": len(returned_before),
      "truncated": before_count > len(returned_before),
      "nearest_predecessor": nearest_predecessor,
      "events": returned_before,
    },
    "overlapping": {
      "matched_count": overlapping_count,
      "returned_count": len(returned_overlapping),
      "truncated": overlapping_count > len(returned_overlapping),
      "events": returned_overlapping,
    },
    "after": {
      "matched_count": after_count,
      "returned_count": len(returned_after),
      "truncated": after_count > len(returned_after),
      "nearest_successor": nearest_successor,
      "events": returned_after,
    },
    "line_hierarchy": {
      "available": hierarchy_unavailable is None,
      "reason": hierarchy_unavailable,
    },
    "ordering": "each partition by (start_ps,end_ps,line_index,event_ordinal); nearest predecessor/successor reported separately",
  }


def _flow_context(
  profile: ProfileRecord, target: EventRecord, limit: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
  # flow 只匹配原始 XStat 标识，不用事件名或时间邻近关系推断因果链。
  if not target.flow:
    return (
      {
        "ready": False,
        "matched_count": 0,
        "returned_count": 0,
        "truncated": False,
        "relations": [],
      },
      diagnostic(
        "FLOW_DATA_UNAVAILABLE",
        "target has no recognized raw XStat flow identifier; temporal results remain valid",
      ),
    )
  target_keys = {_flow_key(flow) for flow in target.flow}
  matched_count = 0
  selected_events: list[EventRecord] = []
  placeholders = ",".join("?" for _ in target_keys)
  with open_profile_database(profile) as connection:
    rows = iter_event_rows(
      connection,
      where_sql=(
        "NOT (e.plane_index = ? AND e.line_index = ? AND e.event_ordinal = ?) "
        f"AND EXISTS (SELECT 1 FROM event_flow AS ef WHERE ef.event_pk = e.event_pk "
        f"AND ef.flow_key IN ({placeholders}))"
      ),
      parameters=(
        target.plane_index,
        target.line_index,
        target.event_ordinal,
        *sorted(target_keys),
      ),
      order_sql="e.plane_index, e.line_index, e.event_ordinal",
    )
    for row in rows:
      matched_count += 1
      if len(selected_events) < limit:
        selected_events.append(hydrate_event_record(connection, row, profile=profile))

  relations: list[dict[str, Any]] = []
  for event in selected_events:
    evidence_count = 0
    returned_evidence: list[dict[str, Any]] = []
    for flow in event.flow:
      if _flow_key(flow) not in target_keys:
        continue
      evidence_count += 1
      if len(returned_evidence) < _FLOW_EVIDENCE_LIMIT:
        returned_evidence.append(flow)
    relations.append(
      {
        "evidence_count": evidence_count,
        "evidence_returned_count": len(returned_evidence),
        "evidence_truncated": len(returned_evidence) < evidence_count,
        "evidence": [_bounded_value(flow) for flow in returned_evidence],
        "evidence_source": "raw_xstat",
        "event": event.to_dict(),
      }
    )
  return (
    {
      "ready": True,
      "scope": "same-profile raw XStat identifiers",
      "matched_count": matched_count,
      "returned_count": len(relations),
      "truncated": matched_count > len(relations),
      "relations": relations,
    },
    None,
  )


def query_context(profile: ProfileRecord, args: Any) -> ContextResult:
  """Resolve one target and compute requested temporal and flow neighborhoods."""
  # 各轴独立统计 matched/returned/truncated，最后再汇总到公共结果。
  target, warnings = _resolve_target(profile, args)
  axis = getattr(args, "axis", "both")
  overlap_limit = getattr(args, "overlap_limit", 100)
  vertical = (
    _vertical_context(profile, target, overlap_limit)
    if axis in {"both", "vertical"}
    else None
  )
  horizontal = (
    _horizontal_context(
      profile,
      target,
      before_limit=getattr(args, "before", 5),
      after_limit=getattr(args, "after", 5),
      window_before_ps=getattr(args, "window_before_ps", None),
      window_after_ps=getattr(args, "window_after_ps", None),
      overlap_limit=overlap_limit,
    )
    if axis in {"both", "horizontal"}
    else None
  )
  flow_related = None
  if getattr(args, "include_flow", False):
    flow_related, flow_warning = _flow_context(profile, target, overlap_limit)
    if flow_warning is not None:
      warnings.append(flow_warning)

  matched = 0
  returned = 0
  truncated = False
  if vertical is not None:
    matched += vertical["matched_count"]
    returned += vertical["returned_count"]
    truncated = truncated or vertical["truncated"]
  if horizontal is not None:
    for section in ("before", "overlapping", "after"):
      matched += horizontal[section]["matched_count"]
      returned += horizontal[section]["returned_count"]
      truncated = truncated or horizontal[section]["truncated"]
  if flow_related is not None:
    matched += flow_related["matched_count"]
    returned += flow_related["returned_count"]
    truncated = truncated or flow_related["truncated"]
  return ContextResult(
    target=target,
    vertical=vertical,
    horizontal=horizontal,
    flow_related=flow_related,
    warnings=tuple(warnings),
    matched_count=matched,
    returned_count=returned,
    truncated=truncated,
    ordering="vertical overlap rank; horizontal nearest/time rank; flow stable locator",
  )


# ============================================================================
# Command-line interface
# ============================================================================

# CLI 成功时只向 stdout 写一个版本化 JSON envelope；预期输入/缓存/查询错误只向
# stderr 写一行并返回 2。BrokenPipe 表示下游提前关闭，不等价于完整输出成功。

_DIAGNOSTIC_NESTED_LIMIT = 10
_DIAGNOSTIC_MAX_DEPTH = 8
_GROUP_BY_HELP = (
  "Comma-separated grouping fields (default: plane,line,event).\n"
  "  plane: plane, plane-id, plane-index, plane-name\n"
  "  line: line, line-id, line-index, line-name, line-display-name\n"
  "  event: event, event-name, event-display-name, metadata-id\n"
  "  XStat: stat, stat-name, stat-type, stat-value\n"
  "  other: kind, hlo, hlo-op, hlo-module, hlo-category\n"
  "Composite plane/line/event/stat/hlo fields retain full parent scope.\n"
  "Granular *-name/*-id fields deliberately aggregate across parent scopes."
)
_METRICS_HELP = (
  "Comma-separated metrics (default: all applicable metrics):\n"
  "  duration, interval, gap, concurrency, self-time, xstats\n"
  "Inapplicable metrics report a reason."
)


def _bounded_strings(values: tuple[str, ...]) -> list[str]:
  return [_truncate_text(value) for value in values[:_PUBLIC_LIST_LIMIT]]


def _bounded_diagnostic_value(value: Any, *, depth: int = 0) -> Any:
  """Recursively bound untrusted diagnostic details for success and errors."""
  # 诊断也可能携带生产者原文，因此使用独立的深度和元素上限后再进入 envelope。
  if depth >= _DIAGNOSTIC_MAX_DEPTH:
    return {"truncated": True, "reason": "MAX_DIAGNOSTIC_DEPTH"}
  if isinstance(value, str):
    if len(value) <= _PUBLIC_TEXT_LIMIT:
      return value
    raw = value.encode("utf-8", errors="surrogatepass")
    return {
      "value_type": "string",
      "value": _truncate_text(value),
      "truncated": True,
      "size_bytes": len(raw),
      "sha256": hashlib.sha256(raw).hexdigest(),
    }
  if isinstance(value, dict):
    items = list(value.items())
    bounded: dict[str, Any] = {}
    for key, item in items[:_DIAGNOSTIC_NESTED_LIMIT]:
      bounded_key = _truncate_text(str(key))
      if isinstance(item, str):
        bounded[bounded_key] = _truncate_text(item)
        if len(item) > _PUBLIC_TEXT_LIMIT:
          raw = item.encode("utf-8", errors="surrogatepass")
          bounded[f"{bounded_key}_truncated"] = True
          bounded[f"{bounded_key}_size_bytes"] = len(raw)
          bounded[f"{bounded_key}_sha256"] = hashlib.sha256(raw).hexdigest()
      else:
        bounded[bounded_key] = _bounded_diagnostic_value(item, depth=depth + 1)
    if len(items) <= _DIAGNOSTIC_NESTED_LIMIT:
      return bounded
    return {
      "value_type": "object",
      "count": len(items),
      "returned_count": min(len(items), _DIAGNOSTIC_NESTED_LIMIT),
      "truncated": True,
      "entries": bounded,
    }
  if isinstance(value, (list, tuple)):
    items = [
      _bounded_diagnostic_value(item, depth=depth + 1)
      for item in value[:_DIAGNOSTIC_NESTED_LIMIT]
    ]
    if len(value) <= _DIAGNOSTIC_NESTED_LIMIT:
      return items
    return {
      "value_type": "array",
      "count": len(value),
      "returned_count": len(items),
      "truncated": True,
      "items": items,
    }
  if value is None or isinstance(value, (bool, int, float)):
    return value
  return _truncate_text(repr(value))


class _ArgumentParser(argparse.ArgumentParser):
  def error(self, message: str) -> None:
    raise CLIError("INVALID_ARGUMENT", message)


def _cli_nonnegative_int(raw: str) -> int:
  try:
    value = int(raw)
  except ValueError as error:
    raise argparse.ArgumentTypeError(f"expected an integer: {raw!r}") from error
  if value < 0:
    raise argparse.ArgumentTypeError(f"expected a nonnegative integer: {raw!r}")
  return value


def _positive_int(raw: str) -> int:
  value = _cli_nonnegative_int(raw)
  if value == 0:
    raise argparse.ArgumentTypeError(f"expected a positive integer: {raw!r}")
  return value


def _half_open_index_range(raw: str, *, subject: str) -> tuple[int | None, int | None]:
  """Parse a zero-based, half-open index range such as ``100:200``."""
  parts = raw.split(":")
  if len(parts) != 2:
    raise argparse.ArgumentTypeError(
      f"{subject} range must use START:STOP syntax, for example 100:200"
    )

  def bound(value: str) -> int | None:
    value = value.strip()
    return None if not value else _cli_nonnegative_int(value)

  start, stop = (bound(part) for part in parts)
  if start is not None and stop is not None and start > stop:
    raise argparse.ArgumentTypeError(
      f"{subject} range start must not exceed stop: {raw!r}"
    )
  return start, stop


def _plane_index_range(raw: str) -> tuple[int | None, int | None]:
  return _half_open_index_range(raw, subject="plane")


def _line_index_range(raw: str) -> tuple[int | None, int | None]:
  return _half_open_index_range(raw, subject="line")


_DURATION_RE = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(ps|ns|us|ms|s)$")
_DURATION_MULTIPLIERS = {
  "ps": Decimal(1),
  "ns": Decimal(1_000),
  "us": Decimal(1_000_000),
  "ms": Decimal(1_000_000_000),
  "s": Decimal(1_000_000_000_000),
}


def _duration_ps(raw: str) -> int:
  match = _DURATION_RE.fullmatch(raw.strip())
  if match is None:
    raise argparse.ArgumentTypeError(
      "duration must use ps, ns, us, ms, or s (for example 100us)"
    )
  try:
    value = Decimal(match.group(1)) * _DURATION_MULTIPLIERS[match.group(2)]
  except InvalidOperation as error:
    raise argparse.ArgumentTypeError(f"invalid duration: {raw!r}") from error
  integral = value.to_integral_value()
  if value != integral:
    raise argparse.ArgumentTypeError(
      "duration must resolve to an integer number of picoseconds"
    )
  if integral < 0:
    raise argparse.ArgumentTypeError("duration must be nonnegative")
  return int(integral)


def _add_common(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("profile", type=Path, metavar="PROFILE.pb")


def _append(parser: argparse.ArgumentParser, option: str, **kwargs: Any) -> None:
  parser.add_argument(option, action="append", **kwargs)


def _add_selectors(parser: argparse.ArgumentParser) -> None:
  _append(
    parser,
    "--event-id",
    dest="event_id",
    metavar="LOCATOR",
    help="PB-local locator: plane:<N>/line:<N>/event:<N>",
  )
  _append(parser, "--plane", metavar="NAME")
  _append(parser, "--plane-id", type=int, metavar="ID")
  _append(parser, "--plane-index", type=_cli_nonnegative_int, metavar="N")
  _append(parser, "--line", metavar="NAME")
  _append(parser, "--line-id", type=int, metavar="ID")
  _append(parser, "--line-index", type=_cli_nonnegative_int, metavar="N")
  _append(parser, "--event", metavar="NAME")
  _append(parser, "--metadata-id", type=int, metavar="ID")
  _append(parser, "--stat", metavar="NAME")
  _append(parser, "--stat-value", metavar="JSON_VALUE")
  parser.add_argument("--start-ps", type=int)
  parser.add_argument("--end-ps", type=int)
  parser.add_argument(
    "--time-relation", choices=("overlap", "contained", "starts-in"), default="overlap"
  )
  parser.add_argument("--min-duration-ps", type=int)
  parser.add_argument("--max-duration-ps", type=int)
  _append(parser, "--kind", choices=("span", "instant", "aggregate", "untimed"))
  _append(parser, "--hlo-op", metavar="NAME")
  parser.add_argument("--match", choices=("exact", "glob", "regex"), default="exact")


def build_parser() -> argparse.ArgumentParser:
  """Build the four-command public parser; no legacy mode is registered."""
  # 这里只注册当前公开的四个子命令，避免旧参数被 argparse 静默接受。
  parser = _ArgumentParser(
    prog="xprof-cli.py",
    description="Query raw tensorflow.profiler.XSpace events with stable locators and integer-picosecond semantics.",
  )
  parser.add_argument(
    "--version",
    action="version",
    version=f"%(prog)s {CLI_SCHEMA_VERSION} (XProf boundary {XPROF_VERSION})",
  )
  subcommands = parser.add_subparsers(
    dest="command", required=True, parser_class=_ArgumentParser
  )

  inspect_parser = subcommands.add_parser(
    "inspect", help="validate one input and inspect structure/cache readiness"
  )
  _add_common(inspect_parser)
  inspect_parser.add_argument(
    "--planes",
    dest="plane_range",
    type=_plane_index_range,
    default=(None, None),
    metavar="START:STOP",
    help=(
      "display planes whose zero-based plane_index is in the half-open range "
      "[START, STOP); omit either bound with :STOP or START: (default: all); "
      "the total plane count is always reported"
    ),
  )
  inspect_parser.add_argument(
    "--plane-index",
    type=_cli_nonnegative_int,
    metavar="N",
    help="display line summaries for the plane with this unique physical index",
  )
  inspect_parser.add_argument(
    "--lines",
    dest="line_range",
    type=_line_index_range,
    metavar="START:STOP",
    help=(
      "display selected-plane lines whose zero-based line_index is in the "
      "half-open range [START, STOP); default: all lines"
    ),
  )

  events_parser = subcommands.add_parser(
    "events", help="discover and page normalized EventRecords"
  )
  _add_common(events_parser)
  _add_selectors(events_parser)
  events_parser.add_argument(
    "--sort", choices=("time", "duration", "name"), default="time"
  )
  events_parser.add_argument(
    "--limit",
    type=_positive_int,
    default=100,
    metavar="N",
    help="maximum EventRecords in this page (default: 100); filtering is unchanged",
  )
  events_parser.add_argument(
    "--cursor",
    metavar="TOKEN",
    help="opaque next-page token from a prior events query on the same profile and options",
  )

  stats_parser = subcommands.add_parser(
    "stats",
    help="group and aggregate selected events",
    formatter_class=argparse.RawTextHelpFormatter,
  )
  _add_common(stats_parser)
  _add_selectors(stats_parser)
  stats_parser.add_argument("--group-by", metavar="FIELDS", help=_GROUP_BY_HELP)
  stats_parser.add_argument("--metrics", metavar="METRICS", help=_METRICS_HELP)
  stats_parser.add_argument(
    "--percentile-mode",
    choices=("exact", "approximate"),
    default="exact",
    help=(
      "percentile algorithm (default: exact)\n"
      "approximate reports its algorithm and error/compression parameters"
    ),
  )
  stats_parser.add_argument(
    "--scan-limit",
    type=_cli_nonnegative_int,
    metavar="N",
    help=(
      "evaluate at most N indexed candidates in stable scan order\n"
      "any explicit value marks scan_complete=false"
    ),
  )
  stats_parser.add_argument(
    "--limit",
    type=_positive_int,
    default=100,
    metavar="N",
    help=("maximum group rows in this page (default: 100)\naggregation is unchanged"),
  )
  stats_parser.add_argument(
    "--cursor",
    metavar="TOKEN",
    help=(
      "opaque next-page token from a prior stats query\n"
      "requires the same profile and options"
    ),
  )

  context_parser = subcommands.add_parser(
    "context", help="query temporal and optional flow context for one event"
  )
  _add_common(context_parser)
  _add_selectors(context_parser)
  context_parser.add_argument(
    "--axis", choices=("both", "vertical", "horizontal"), default="both"
  )
  context_parser.add_argument("--before", type=_cli_nonnegative_int, default=5)
  context_parser.add_argument("--after", type=_cli_nonnegative_int, default=5)
  context_parser.add_argument(
    "--window-before", dest="window_before_ps", type=_duration_ps
  )
  context_parser.add_argument(
    "--window-after", dest="window_after_ps", type=_duration_ps
  )
  context_parser.add_argument("--overlap-limit", type=_cli_nonnegative_int, default=100)
  context_parser.add_argument("--include-flow", action="store_true")
  return parser


def _read_selected_plane(
  connection: sqlite3.Connection,
  *,
  plane_index: int | None,
  line_range: tuple[int | None, int | None],
) -> dict[str, Any] | None:
  """Resolve one plane and return its optionally ranged line summaries."""
  if plane_index is None:
    return None

  plane = connection.execute(
    """
    SELECT p.plane_index, p.plane_id, p.name, p.line_count,
           coalesce(sum(l.event_count), 0) AS event_count
    FROM plane AS p
    LEFT JOIN line AS l ON l.plane_index = p.plane_index
    WHERE p.plane_index = ?
    GROUP BY p.plane_index, p.plane_id, p.name, p.line_count
    """,
    (plane_index,),
  ).fetchone()
  if plane is None:
    raise CLIError("PLANE_NOT_FOUND", f"no plane matched plane_index={plane_index}")

  line_start, line_stop = line_range
  lines = tuple(
    {
      "line_index": row["line_index"],
      "line_id": row["line_id"],
      "display_id": row["display_id"],
      "name": row["name"],
      "display_name": row["display_name"],
      "timestamp_ns": row["timestamp_ns"],
      "duration_ps": row["duration_ps"],
      "event_count": row["event_count"],
      "event_kind_counts": {
        kind: row[f"{kind}_count"]
        for kind in ("span", "instant", "aggregate", "untimed")
      },
    }
    for row in connection.execute(
      """
      SELECT line_index, line_id, display_id, name, display_name,
             timestamp_ns, duration_ps, event_count,
             span_count, instant_count, aggregate_count, untimed_count
      FROM line
      WHERE plane_index = ?
        AND (? IS NULL OR line_index >= ?)
        AND (? IS NULL OR line_index < ?)
      ORDER BY line_index
      """,
      (
        plane["plane_index"],
        line_start,
        line_start,
        line_stop,
        line_stop,
      ),
    )
  )
  return {
    "plane_index": plane["plane_index"],
    "plane_id": plane["plane_id"],
    "name": plane["name"],
    "line_count": plane["line_count"],
    "event_count": plane["event_count"],
    "lines": lines,
  }


def _read_profile_metadata(
  profile: ProfileRecord,
  *,
  include_structure: bool = False,
  plane_range: tuple[int | None, int | None] = (None, None),
  selected_plane_index: int | None = None,
  line_range: tuple[int | None, int | None] = (None, None),
) -> dict[str, Any]:
  """Query envelope metadata plus optional plane and line summary ranges."""
  # 总数直接查询完整 profile；两个 range 只约束 inspect 实际展示的结构行。
  with open_profile_database(profile) as connection:
    profile_row = connection.execute(
      """
      SELECT span_count, instant_count, aggregate_count, untimed_count,
             validation_json
      FROM profile WHERE singleton = 1
      """
    ).fetchone()
    if profile_row is None:
      raise RuntimeError("SQLite profile metadata is missing")
    validation = json.loads(profile_row["validation_json"])
    if not isinstance(validation, dict):
      raise ValueError("SQLite validation metadata is not an object")
    validation = dict(validation)
    validation["validation_source"] = (
      "cache-validated" if profile.cache.get("status") == "hit" else "checked"
    )

    def values(query: str, parameters: tuple[Any, ...] = ()) -> tuple[str, ...]:
      return tuple(
        row[0]
        for row in connection.execute(
          query + " LIMIT ?", (*parameters, _PUBLIC_LIST_LIMIT + 1)
        )
      )

    def count(query: str, parameters: tuple[Any, ...] = ()) -> int:
      row = connection.execute(query, parameters).fetchone()
      assert row is not None
      return int(row[0])

    hostnames = values("SELECT value FROM hostname ORDER BY ordinal")
    hostname_count = count("SELECT count(*) FROM hostname")
    capture_errors = values(
      "SELECT value FROM capture_message WHERE kind = ? ORDER BY ordinal",
      ("error",),
    )
    capture_error_count = count(
      "SELECT count(*) FROM capture_message WHERE kind = ?", ("error",)
    )
    capture_warnings = values(
      "SELECT value FROM capture_message WHERE kind = ? ORDER BY ordinal",
      ("warning",),
    )
    capture_warning_count = count(
      "SELECT count(*) FROM capture_message WHERE kind = ?", ("warning",)
    )

    diagnostics: list[dict[str, Any]] = []
    for row in connection.execute(
      """
      SELECT code, message, details_json
      FROM diagnostic ORDER BY ordinal LIMIT ?
      """,
      (_PUBLIC_LIST_LIMIT,),
    ):
      details = json.loads(row["details_json"])
      if not isinstance(details, dict):
        raise ValueError("SQLite diagnostic details are not an object")
      diagnostics.append({"code": row["code"], "message": row["message"], **details})
    diagnostic_count = count("SELECT count(*) FROM diagnostic")
    if hostname_count > 1:
      diagnostics.extend(
        (
          diagnostic(
            "HOST_ASSIGNMENT_AMBIGUOUS",
            "XSpace lists multiple hostnames but has no plane-to-host mapping; events are not assigned to an arbitrary host and cross-line clocks remain isolated",
            hostname_count=hostname_count,
          ),
          diagnostic(
            "CLOCK_ALIGNMENT_UNKNOWN",
            "cross-line timestamps are not compared because a multi-host XSpace has no plane-to-host clock evidence",
            hostname_count=hostname_count,
          ),
        )
      )
      diagnostic_count += 2

    event_kind_counts = {
      kind: int(profile_row[f"{kind}_count"])
      for kind in ("span", "instant", "aggregate", "untimed")
    }
    result: dict[str, Any] = {
      "hostnames": hostnames,
      "hostname_count": hostname_count,
      "capture_errors": capture_errors,
      "capture_error_count": capture_error_count,
      "capture_warnings": capture_warnings,
      "capture_warning_count": capture_warning_count,
      "diagnostics": tuple(diagnostics),
      "diagnostic_count": diagnostic_count,
      "validation": validation,
      "plane_count": count("SELECT count(*) FROM plane"),
      "line_count": count("SELECT count(*) FROM line"),
      "event_count": sum(event_kind_counts.values()),
      "event_kind_counts": event_kind_counts,
    }
    if not include_structure:
      return result

    plane_start, plane_stop = plane_range
    result["planes"] = tuple(
      {
        "plane_index": row["plane_index"],
        "plane_id": row["plane_id"],
        "name": row["name"],
        "line_count": row["line_count"],
        "event_count": row["event_count"],
      }
      for row in connection.execute(
        """
        SELECT p.plane_index, p.plane_id, p.name, p.line_count,
               coalesce(sum(l.event_count), 0) AS event_count
        FROM plane AS p
        LEFT JOIN line AS l ON l.plane_index = p.plane_index
        WHERE (? IS NULL OR p.plane_index >= ?)
          AND (? IS NULL OR p.plane_index < ?)
        GROUP BY p.plane_index, p.plane_id, p.name, p.line_count
        ORDER BY p.plane_index
        """,
        (plane_start, plane_start, plane_stop, plane_stop),
      )
    )
    result["selected_plane"] = _read_selected_plane(
      connection,
      plane_index=selected_plane_index,
      line_range=line_range,
    )
    return result


def _bounded_warnings(
  profile: ProfileRecord,
  extra: tuple[dict[str, Any], ...] = (),
  limit: int = 100,
  *,
  metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
  metadata = metadata if metadata is not None else _read_profile_metadata(profile)
  total_count = (
    metadata["diagnostic_count"]
    + metadata["capture_error_count"]
    + metadata["capture_warning_count"]
    + len(extra)
  )
  warnings: list[dict[str, Any]] = []

  def append_existing(source: tuple[dict[str, Any], ...]) -> None:
    remaining = limit - len(warnings)
    if remaining > 0:
      warnings.extend(source[:remaining])

  append_existing(extra)
  for code, producer_severity, messages in (
    ("XSPACE_CAPTURE_ERROR", "error", metadata["capture_errors"]),
    ("XSPACE_CAPTURE_WARNING", "warning", metadata["capture_warnings"]),
  ):
    for message in messages:
      if len(warnings) >= limit:
        break
      warnings.append(
        diagnostic(
          code,
          _truncate_text(message),
          location=profile.source_path,
          producer_severity=producer_severity,
          message_truncated=len(message) > _PUBLIC_TEXT_LIMIT,
        )
      )
  append_existing(metadata["diagnostics"])
  if total_count <= limit:
    return [_bounded_diagnostic_value(warning) for warning in warnings]
  bounded = [
    *warnings[: max(0, limit - 1)],
    diagnostic(
      "WARNINGS_TRUNCATED",
      "invocation warnings were bounded in the public envelope",
      total_count=total_count,
      returned_count=limit,
    ),
  ]
  return [_bounded_diagnostic_value(warning) for warning in bounded]


def _common_envelope(
  profile: ProfileRecord,
  command: str,
  *,
  metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
  # 每个命令复用同一 provenance/completeness 骨架，防止某条路径遗漏单位或版本。
  metadata = metadata if metadata is not None else _read_profile_metadata(profile)
  return {
    "schema_version": CLI_SCHEMA_VERSION,
    "command": command,
    "analysis_complete": True,
    "input": {
      "path": profile.source_path,
      "profile_sha256": profile.profile_sha256,
      "raw_size_bytes": profile.raw_size_bytes,
      "validation": metadata["validation"],
    },
    "profile_sha256": profile.profile_sha256,
    "profile_sha256_complete": True,
    "raw_size_bytes": profile.raw_size_bytes,
    "raw_size_bytes_complete": True,
    "cache": dict(profile.cache),
    "parser": {
      "message": PARSER_MESSAGE,
      "schema_revision": PARSER_SCHEMA_REVISION,
      "schema_sha256": PARSER_SCHEMA_SHA256,
      "upstream_schema_sha256": PARSER_SCHEMA_SHA256,
      "check_version": CHECK_VERSION,
      "xprof_boundary_version": XPROF_VERSION,
    },
    "provenance": {
      "raw_events": "pinned OpenXLA xplane.proto compiled in the cache root",
      "official_xprof_reused": False,
      "enrichment": "optional verified raw XStats only",
      "hosts": {
        "source": "pinned_xspace_schema:XSpace.hostnames",
        "official_get_hosts_reused": False,
        "official_api": "get_hosts",
        "official_version": XPROF_VERSION,
        "reason": "OSS_2_23_1_GET_HOSTS_DERIVES_SESSION_FILENAME_LABELS",
      },
    },
    "time_unit": "picoseconds",
    "sorting": None,
    "truncated": False,
    "warnings": _bounded_warnings(profile, metadata=metadata),
    "errors": [],
  }


def _inspect_envelope(
  profile: ProfileRecord, args: argparse.Namespace
) -> dict[str, Any]:
  plane_start, plane_stop = getattr(args, "plane_range", (None, None))
  line_start, line_stop = getattr(args, "line_range", None) or (None, None)
  metadata = _read_profile_metadata(
    profile,
    include_structure=True,
    plane_range=(plane_start, plane_stop),
    selected_plane_index=getattr(args, "plane_index", None),
    line_range=(line_start, line_stop),
  )
  envelope = _common_envelope(profile, "inspect", metadata=metadata)
  plane_rows = [
    {
      "plane_index": plane["plane_index"],
      "plane_id": plane["plane_id"],
      "name": _truncate_text(plane["name"]),
      "name_truncated": len(plane["name"]) > _PUBLIC_TEXT_LIMIT,
      "line_count": plane["line_count"],
      "event_count": plane["event_count"],
    }
    for plane in metadata["planes"]
  ]
  planes_truncated = metadata["plane_count"] != len(plane_rows)
  selected_plane = metadata["selected_plane"]
  selected_plane_payload: dict[str, Any] | None = None
  selected_lines_truncated = False
  if selected_plane is not None:
    lines = [
      {
        **line,
        "name": _truncate_text(line["name"]),
        "name_truncated": len(line["name"]) > _PUBLIC_TEXT_LIMIT,
        "display_name": _truncate_text(line["display_name"]),
        "display_name_truncated": (len(line["display_name"]) > _PUBLIC_TEXT_LIMIT),
      }
      for line in selected_plane["lines"]
    ]
    selected_lines_truncated = selected_plane["line_count"] != len(lines)
    selected_plane_payload = {
      "plane_index": selected_plane["plane_index"],
      "plane_id": selected_plane["plane_id"],
      "name": _truncate_text(selected_plane["name"]),
      "name_truncated": len(selected_plane["name"]) > _PUBLIC_TEXT_LIMIT,
      "line_count": selected_plane["line_count"],
      "event_count": selected_plane["event_count"],
      "line_range": {
        "start": line_start,
        "stop": line_stop,
        "semantics": "zero-based half-open line_index range [start, stop)",
      },
      "lines": lines,
      "lines_returned_count": len(lines),
      "lines_truncated": selected_lines_truncated,
    }
  hostnames = metadata["hostnames"]
  capture_errors = metadata["capture_errors"]
  capture_warnings = metadata["capture_warnings"]
  ambiguous_host_mapping = metadata["hostname_count"] > 1
  envelope.update(
    {
      "profile": {
        "schema_message": PARSER_MESSAGE,
        "hostnames": _bounded_strings(hostnames),
        "hostnames_truncated": metadata["hostname_count"] > _PUBLIC_LIST_LIMIT,
        "hostname_values_truncated": any(
          len(value) > _PUBLIC_TEXT_LIMIT for value in hostnames[:_PUBLIC_LIST_LIMIT]
        ),
        "capture_errors": _bounded_strings(capture_errors),
        "capture_errors_truncated": (
          metadata["capture_error_count"] > _PUBLIC_LIST_LIMIT
        ),
        "capture_error_values_truncated": any(
          len(value) > _PUBLIC_TEXT_LIMIT
          for value in capture_errors[:_PUBLIC_LIST_LIMIT]
        ),
        "capture_warnings": _bounded_strings(capture_warnings),
        "capture_warnings_truncated": (
          metadata["capture_warning_count"] > _PUBLIC_LIST_LIMIT
        ),
        "capture_warning_values_truncated": any(
          len(value) > _PUBLIC_TEXT_LIMIT
          for value in capture_warnings[:_PUBLIC_LIST_LIMIT]
        ),
        "counts": {
          "planes": metadata["plane_count"],
          "lines": metadata["line_count"],
          "events": metadata["event_count"],
        },
        "event_kind_counts": metadata["event_kind_counts"],
        "plane_range": {
          "start": plane_start,
          "stop": plane_stop,
          "semantics": "zero-based half-open plane_index range [start, stop)",
        },
        "planes": plane_rows,
        "planes_returned_count": len(plane_rows),
        "planes_truncated": planes_truncated,
      },
      "clock_alignment": {
        "status": (
          "unknown-within-profile" if ambiguous_host_mapping else "profile-local"
        ),
        "cross_line_timestamps_comparable": not ambiguous_host_mapping,
      },
      "sorting": "physical plane index; selected-plane lines by physical line index",
      "truncated": planes_truncated or selected_lines_truncated,
    }
  )
  if selected_plane_payload is not None:
    envelope["profile"]["selected_plane"] = selected_plane_payload
  return envelope


def _events_envelope(
  profile: ProfileRecord, args: argparse.Namespace
) -> dict[str, Any]:
  result = query_event_page(profile, args)
  metadata = _read_profile_metadata(profile)
  envelope = _common_envelope(profile, "events", metadata=metadata)
  envelope["warnings"] = _bounded_warnings(profile, result.warnings, metadata=metadata)
  envelope.update(
    {
      "matched_count": result.matched_count,
      "returned_count": len(result.items),
      "truncated": result.truncated,
      "next_cursor": result.next_cursor,
      "sorting": result.ordering,
      "events": [event.to_dict() for event in result.items],
    }
  )
  return envelope


def _stats_envelope(profile: ProfileRecord, args: argparse.Namespace) -> dict[str, Any]:
  result = compute_stats(profile, args)
  metadata = _read_profile_metadata(profile)
  envelope = _common_envelope(profile, "stats", metadata=metadata)
  envelope["warnings"] = _bounded_warnings(profile, result.warnings, metadata=metadata)
  envelope.update(
    {
      "matched_count": result.matched_count,
      "returned_count": result.returned_count,
      "truncated": result.truncated,
      "next_cursor": result.next_cursor,
      "sorting": result.ordering,
      "scan_complete": result.scan_complete,
      "scanned_event_count": result.scanned_event_count,
      "selected_event_count": result.selected_event_count,
      "approximation": result.approximation,
      "groups": list(result.groups),
    }
  )
  return envelope


def _context_envelope(
  profile: ProfileRecord, args: argparse.Namespace
) -> dict[str, Any]:
  result = query_context(profile, args)
  metadata = _read_profile_metadata(profile)
  envelope = _common_envelope(profile, "context", metadata=metadata)
  envelope["warnings"] = _bounded_warnings(profile, result.warnings, metadata=metadata)
  envelope.update(
    {
      "matched_count": result.matched_count,
      "returned_count": result.returned_count,
      "truncated": result.truncated,
      "sorting": result.ordering,
      "target": result.target.to_dict(),
      "vertical": result.vertical,
      "horizontal": result.horizontal,
      "flow_related": result.flow_related,
    }
  )
  return envelope


def execute(args: argparse.Namespace) -> dict[str, Any]:
  """Execute a parsed invocation and return its complete public envelope."""
  # 四个命令共享同一个单 profile 装载与缓存边界。
  if (
    args.command == "inspect"
    and getattr(args, "line_range", None) is not None
    and getattr(args, "plane_index", None) is None
  ):
    raise CLIError("INVALID_ARGUMENT", "--lines requires --plane-index")
  profile = load_input_profile(args.profile)
  builders = {
    "inspect": lambda: _inspect_envelope(profile, args),
    "events": lambda: _events_envelope(profile, args),
    "stats": lambda: _stats_envelope(profile, args),
    "context": lambda: _context_envelope(profile, args),
  }
  return builders[args.command]()


def main(argv: list[str] | None = None) -> int:
  """Run the CLI and return a process exit status."""
  # JSON 禁止 NaN；CLIError 走 argparse 的稳定退出格式，意外异常保留 traceback。
  parser = build_parser()
  try:
    args = parser.parse_args(argv)
    envelope = execute(args)
    json.dump(envelope, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0
  except CLIError as error:
    parser.exit(2, f"xprof-cli: {error}\n")
  except BrokenPipeError:
    return 0


if __name__ == "__main__":
  raise SystemExit(main())
