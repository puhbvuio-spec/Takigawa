
from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import xml.etree.ElementTree as ET

import aiohttp
import pandas as pd


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_console_encoding()


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_json_dump(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_hash(text: str, prefix: str = "ID", length: int = 16) -> str:
    return f"{prefix}_{hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:length]}"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def strip_think_blocks(text: str) -> str:
    """移除 qwen3 系列模型可能返回的 <think>...</think> 深度思考块。"""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
        if "<think>" in cleaned:
            cleaned = cleaned[:cleaned.index("<think>")].strip()
    return cleaned or text


def extract_first_balanced_json(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = str(text).strip().replace("\ufeff", "")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    start_idx = None
    for i, ch in enumerate(cleaned):
        if ch in "[{":
            start_idx = i
            break
    if start_idx is None:
        return None
    stack: List[str] = []
    in_str = False
    escape = False
    for j in range(start_idx, len(cleaned)):
        ch = cleaned[j]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return None
            if not stack:
                return cleaned[start_idx:j + 1]
    return None


def robust_json_loads(maybe_json: str) -> Any:
    if not maybe_json:
        return None
    text = re.sub(r",(\s*[}\]])", r"\1", str(maybe_json).strip())
    for candidate in [text, extract_first_balanced_json(text)]:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, str):
                inner = extract_first_balanced_json(obj)
                return robust_json_loads(inner) if inner else obj
            return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            pass
    return None


@dataclass
class RuntimePaths:
    app_root: Path
    data_root: Path
    inbox_dir: Path
    work_root: Path
    runtime_dir: Path
    locks_dir: Path

    @classmethod
    def from_env(cls) -> "RuntimePaths":
        app_root = Path(__file__).resolve().parent
        data_root = Path(os.environ.get("CF_DATA_ROOT", app_root / "data"))
        return cls(
            app_root=app_root,
            data_root=data_root,
            inbox_dir=data_root / "inbox",
            work_root=data_root / "work",
            runtime_dir=data_root / "runtime",
            locks_dir=data_root / "runtime" / "locks",
        )

    def ensure(self) -> None:
        for p in [self.data_root, self.inbox_dir, self.work_root, self.runtime_dir, self.locks_dir]:
            ensure_dir(p)


@dataclass
class PipelineConfig:
    label_library_path: Path
    llm_enabled: bool = False
    api_key: str = ""
    api_base_url: str = ""  # 留空则根据 api_key 前缀自动推断
    model_name: str = "qwen3.5-plus"
    max_concurrency: int = 5
    label_batch_plan: List[int] = field(default_factory=lambda: [200, 100, 50, 20, 5, 1])
    boundary_batch_plan: List[int] = field(default_factory=lambda: [100, 50, 20, 5, 1])
    similarity_threshold: float = 0.92
    enable_near_dup: bool = True
    retriever_topk: int = 8
    review_label_density_threshold: int = 4

    def resolve_api_base_url(self) -> str:
        """根据 api_key 前缀自动推断 API 地址（与 run_ai_tagging_rag.py 保持一致）。"""
        if self.api_base_url:
            return self.api_base_url
        key = self.api_key.strip()
        if key.startswith("sk-sp-"):
            return "https://coding.dashscope.aliyuncs.com/v1/chat/completions"
        return "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def validate(self) -> None:
        if self.llm_enabled and not self.api_key:
            raise RuntimeError("已启用 LLM 模式，但未提供 API KEY。请先设置 DASHSCOPE_API_KEY 再运行。")
        if not self.label_library_path.exists():
            raise FileNotFoundError(f"标签库不存在：{self.label_library_path}")
        # 自动填充 api_base_url
        if self.llm_enabled and not self.api_base_url:
            object.__setattr__(self, "api_base_url", self.resolve_api_base_url())


@dataclass
class RawFeedback:
    record_id: str
    node_id: str
    source_platform: str
    publish_time: str
    post_title: str
    post_tag: List[str]
    raw_text: str
    source_file: str
    source_url: str = ""
    user_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanFeedback:
    record_id: str
    node_id: str
    source_platform: str
    publish_time: str
    post_title: str
    post_tag: List[str]
    raw_text: str
    normalized_text: str
    cleaned_text: str
    source_file: str
    source_url: str = ""
    user_id: str = ""
    is_spam: bool = False
    is_template: bool = False
    duplicate_of: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    candidate_cuts: List[Dict[str, Any]] = field(default_factory=list)
    default_segments: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelDefinition:
    label_id: str
    level_1: str
    level_2: str
    definition: str
    sentiment_enabled: bool = True
    sentiment_guide: Dict[str, str] = field(default_factory=dict)
    anchor_terms: List[str] = field(default_factory=list)
    boundary_hint: str = ""
    conflict_labels: List[str] = field(default_factory=list)
    status: str = "active"


@dataclass
class LabelHit:
    label_id: str
    sentiment: str
    confidence: float = 0.0
    source: str = ""


@dataclass
class LabelResult:
    record_id: str
    hits: List[LabelHit] = field(default_factory=list)
    raw_model_payload: Any = None


@dataclass
class BoundaryResult:
    record_id: str
    decision: str
    cut_ids: List[str] = field(default_factory=list)
    reason: str = ""
    raw_model_payload: Any = None


@dataclass
class JobContext:
    job_id: str
    job_dir: Path
    input_dir: Path
    output_dir: Path
    assets_dir: Path
    reports_dir: Path
    meta_path: Path


class WorkspaceManager:
    @staticmethod
    def is_valid_input_file(path: Path) -> bool:
        return path.is_file() and not path.name.startswith("~$") and path.suffix.lower() in {".csv", ".json", ".jsonl", ".xlsx"}

    def __init__(self, runtime_paths: RuntimePaths):
        self.runtime_paths = runtime_paths

    def prepare_single_job(self) -> JobContext:
        self.runtime_paths.ensure()
        files = sorted([p for p in self.runtime_paths.inbox_dir.iterdir() if self.is_valid_input_file(p)], key=lambda p: p.name)
        if not files:
            raise RuntimeError(f"inbox 中没有可处理文件：{self.runtime_paths.inbox_dir}")
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_dir = self.runtime_paths.work_root / job_id
        input_dir = job_dir / "input"
        output_dir = job_dir / "output"
        assets_dir = output_dir / "assets"
        reports_dir = output_dir / "reports"
        for p in [job_dir, input_dir, output_dir, assets_dir, reports_dir]:
            ensure_dir(p)
        moved = []
        for src in files:
            dst = input_dir / src.name
            shutil.copy2(src, dst)
            moved.append(dst.name)
        meta_path = job_dir / "meta.json"
        safe_json_dump(meta_path, {
            "job_id": job_id,
            "created_at": now_ts(),
            "input_files": moved,
            "status": "prepared",
            "stages": {
                "ingest": "pending",
                "preprocess": "pending",
                "aggregate": "pending",
                "labeling": "pending",
                "boundary_validation": "pending",
                "reporting": "pending",
            }
        })
        return JobContext(job_id, job_dir, input_dir, output_dir, assets_dir, reports_dir, meta_path)


class JobMetaStore:
    @staticmethod
    def update(meta_path: Path, **kwargs: Any) -> None:
        meta = load_json(meta_path) if meta_path.exists() else {}
        meta.update(kwargs)
        safe_json_dump(meta_path, meta)

    @staticmethod
    def update_stage(meta_path: Path, stage_name: str, stage_status: str, error: Optional[str] = None) -> None:
        meta = load_json(meta_path) if meta_path.exists() else {}
        stages = meta.get("stages", {})
        stages[stage_name] = stage_status
        meta["stages"] = stages
        if error is not None:
            meta["error"] = error
        safe_json_dump(meta_path, meta)


class FeedbackLoader:
    REQUIRED_FIELDS = {
        "raw_text": ["raw_text", "text", "评论内容", "comment_text", "content"],
        "node_id": ["node_id", "观测节点", "node", "topic_id"],
        "source_platform": ["source_platform", "platform", "来源平台", "source"],
        "publish_time": ["publish_time", "时间", "发布时间", "submit_time"],
        "post_title": ["post_title", "标题", "title", "source_title"],
        "post_tag": ["post_tag", "标签", "tags", "source_tag"],
        "source_url": ["source_url", "url", "链接"],
        "user_id": ["user_id", "uid", "用户id", "用户ID"],
        "record_id": ["record_id", "评论ID", "comment_id", "id"],
    }

    def load_dir(self, input_dir: Path) -> List[RawFeedback]:
        rows: List[RawFeedback] = []
        for path in sorted(input_dir.iterdir(), key=lambda p: p.name):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                rows.extend(self._rows_from_df(pd.read_csv(path), path.name))
            elif suffix == ".xlsx":
                rows.extend(self._rows_from_df(pd.read_excel(path), path.name))
            elif suffix == ".json":
                payload = load_json(path)
                rows.extend(self._rows_from_list(payload if isinstance(payload, list) else [payload], path.name))
            elif suffix == ".jsonl":
                payloads = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            payloads.append(json.loads(line))
                rows.extend(self._rows_from_list(payloads, path.name))
        if not rows:
            raise RuntimeError(f"未从输入目录读取到任何可处理反馈：{input_dir}")
        return rows

    def _pick(self, row: Dict[str, Any], canonical_key: str, default: Any = "") -> Any:
        for key in self.REQUIRED_FIELDS.get(canonical_key, []):
            if key in row:
                return row.get(key, default)
        return default

    def _normalize_tags(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [normalize_ws(x) for x in value if normalize_ws(x)]
        raw = normalize_ws(value)
        if not raw:
            return []
        return [normalize_ws(x) for x in re.split(r"[;,，|/]", raw) if normalize_ws(x)]

    def _rows_from_df(self, df: pd.DataFrame, source_file: str) -> List[RawFeedback]:
        return self._rows_from_list(df.fillna("").to_dict(orient="records"), source_file)

    def _rows_from_list(self, rows: List[Dict[str, Any]], source_file: str) -> List[RawFeedback]:
        out: List[RawFeedback] = []
        for idx, row in enumerate(rows, start=1):
            raw_text = normalize_ws(self._pick(row, "raw_text"))
            record_id = normalize_ws(self._pick(row, "record_id")) or stable_hash(f"{source_file}|{idx}|{raw_text}", prefix="RID")
            node_id = normalize_ws(self._pick(row, "node_id")) or "unknown_node"
            source_platform = normalize_ws(self._pick(row, "source_platform")) or "unknown_platform"
            publish_time = normalize_ws(self._pick(row, "publish_time")) or "unknown_time"
            post_title = normalize_ws(self._pick(row, "post_title"))
            post_tag = self._normalize_tags(self._pick(row, "post_tag"))
            source_url = normalize_ws(self._pick(row, "source_url"))
            user_id = normalize_ws(self._pick(row, "user_id"))
            known_keys = set()
            for candidates in self.REQUIRED_FIELDS.values():
                known_keys.update(candidates)
            extra = {k: v for k, v in row.items() if k not in known_keys}
            out.append(RawFeedback(record_id, node_id, source_platform, publish_time, post_title, post_tag, raw_text, source_file, source_url, user_id, extra))
        return out


class LabelLibrary:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.labels: List[LabelDefinition] = []
        for item in payload.get("labels", []):
            self.labels.append(LabelDefinition(
                label_id=str(item.get("label_id", "")).strip(),
                level_1=str(item.get("level_1", "")).strip(),
                level_2=str(item.get("level_2", "")).strip(),
                definition=str(item.get("definition", "")).strip(),
                sentiment_enabled=bool(item.get("sentiment_enabled", True)),
                sentiment_guide=item.get("sentiment_guide", {}) or {},
                anchor_terms=[normalize_ws(x) for x in item.get("anchor_terms", []) if normalize_ws(x)],
                boundary_hint=str(item.get("boundary_hint", "")).strip(),
                conflict_labels=[normalize_ws(x) for x in item.get("conflict_labels", []) if normalize_ws(x)],
                status=str(item.get("status", "active")).strip() or "active",
            ))
        self.label_index = {x.label_id: x for x in self.labels}

    @classmethod
    def from_path(cls, path: Path) -> "LabelLibrary":
        return cls(load_json(path))

    def active_labels(self) -> List[LabelDefinition]:
        return [x for x in self.labels if x.status == "active"]


class MindMapNodeMapper:
    """解析 FreeMind .mm 文件，提取次末端节点名称，构建 node_id → 中文名映射。

    次末端节点 = 叶节点的父节点（因为最末端指向的是具体创作者）。
    """

    @staticmethod
    def _clean_text(raw: str) -> str:
        """移除 mm 文件中常见的 HTML 换行标记等噪声。"""
        import html as _html
        t = _html.unescape(_html.unescape(raw))  # 双重 unescape 处理 &amp;lt; 等
        t = re.sub(r"<br\s*/?>", "", t, flags=re.IGNORECASE)
        return t.strip()

    @classmethod
    def _collect_penultimate(cls, node: ET.Element, depth: int = 0) -> List[str]:
        """递归收集所有次末端节点的 TEXT。"""
        children = list(node)
        child_nodes = [c for c in children if c.tag == "node"]
        if not child_nodes:
            # 当前节点是叶节点，不是次末端
            return []
        # 判断是否所有子节点都是叶节点
        all_children_are_leaves = all(
            len([gc for gc in c if gc.tag == "node"]) == 0
            for c in child_nodes
        )
        results: List[str] = []
        if all_children_are_leaves and depth >= 2:
            # 当前节点就是次末端节点
            text = cls._clean_text(node.get("TEXT", ""))
            if text:
                results.append(text)
        else:
            for c in child_nodes:
                results.extend(cls._collect_penultimate(c, depth + 1))
        return results

    @classmethod
    def load(cls, mm_path: Path) -> Dict[str, str]:
        """返回一个 {node_id: 次末端中文名} 的映射字典。

        映射逻辑：将所有次末端节点名称提取出来，与运行时出现的 node_id
        尝试模糊匹配（node_id 通常是 mm 文件中 ID 属性的简化形式，
        或者在数据准备阶段由人工指定）。
        此处返回 {mm_node_ID → 清洁文本} 以及 {清洁文本 lowercase → 清洁文本}
        供上游按需查找。
        """
        if not mm_path.exists():
            return {}
        try:
            tree = ET.parse(mm_path)
        except Exception:
            return {}
        root = tree.getroot()
        # 收集所有次末端节点及其 ID
        mapping: Dict[str, str] = {}
        cls._collect_with_id(root, mapping, depth=0)
        return mapping

    @classmethod
    def _collect_with_id(cls, node: ET.Element, out: Dict[str, str], depth: int) -> None:
        children = [c for c in node if c.tag == "node"]
        if not children:
            return
        all_children_are_leaves = all(
            len([gc for gc in c if gc.tag == "node"]) == 0
            for c in children
        )
        if all_children_are_leaves and depth >= 2:
            text = cls._clean_text(node.get("TEXT", ""))
            node_id = node.get("ID", "")
            if text:
                out[node_id] = text
                # 也存一份小写简化版，便于模糊匹配
                simplified = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", text).lower()
                out[simplified] = text
        else:
            for c in children:
                cls._collect_with_id(c, out, depth + 1)

    @classmethod
    def build_runtime_mapping(cls, mm_path: Path, node_ids: set) -> Dict[str, str]:
        """构建运行时 node_id → 显示名 的映射。

        优先使用 mm 文件中的 ID 精确匹配；
        回退到 node_id 中嵌入的中文片段模糊匹配。
        """
        raw_map = cls.load(mm_path)
        if not raw_map:
            return {}
        # 提取所有 (mm_id, display_name) 对
        id_to_name: Dict[str, str] = {}
        for k, v in raw_map.items():
            if k.startswith("ID_"):
                id_to_name[k] = v

        # 构建反向索引：display_name → display_name（自身映射，用于中文 node_id 直接命中）
        all_display_names = set(id_to_name.values())

        result: Dict[str, str] = {}
        for nid in node_ids:
            # 1. 精确匹配 mm ID
            if nid in id_to_name:
                result[nid] = id_to_name[nid]
                continue
            # 2. node_id 本身就是中文次末端名称
            cleaned_nid = cls._clean_text(nid)
            if cleaned_nid in all_display_names:
                result[nid] = cleaned_nid
                continue
            # 3. 模糊匹配：node_id 中包含中文关键词
            best_match = None
            best_score = 0
            nid_lower = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", nid).lower()
            for display_name in all_display_names:
                dn_lower = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", display_name).lower()
                # 检查子串包含关系
                if nid_lower and dn_lower and (nid_lower in dn_lower or dn_lower in nid_lower):
                    score = len(dn_lower)
                    if score > best_score:
                        best_score = score
                        best_match = display_name
            if best_match:
                result[nid] = best_match
            # 否则保持原 node_id，不做映射
        return result


class FeedbackPreprocessor:
    URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    AT_RE = re.compile(r"@[^\s，。！!？?,]+")
    PHONE_RE = re.compile(r"1\d{10}")
    WECHAT_RE = re.compile(r"(?:vx|Vx|VX|wx|WX|微信)[:：]?\s*[a-zA-Z0-9_-]{5,}")
    HTML_RE = re.compile(r"<[^>]+>")
    TS_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?\b")
    MULTI_SPACE_RE = re.compile(r"\s+")
    REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
    SENTENCE_CUT_RE = re.compile(r"[，,。；;！!？?]|不过|但是|但|然而|而且|以及")
    TEMPLATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"转载", r"来源[:：]", r"作者[:：]", r"侵删", r"免责声明", r"欢迎关注", r"点赞关注"]]
    SPAM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"加微", r"私聊", r"返利", r"代充", r"兼职", r"博彩", r"色情", r"带你赚钱"]]
    INVALID_SET = {"", "nan", "none", "null", "nil", "n/a", "na", "-", "--", "无", "没有", "沒", "no", "nothing"}

    def __init__(self, similarity_threshold: float = 0.92, enable_near_dup: bool = True):
        self.similarity_threshold = similarity_threshold
        self.enable_near_dup = enable_near_dup

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text).replace("\u3000", " ")
        return self.MULTI_SPACE_RE.sub(" ", text).strip()

    def strip_noise(self, text: str) -> str:
        text = self.URL_RE.sub(" ", text)
        text = self.AT_RE.sub(" ", text)
        text = self.PHONE_RE.sub(" ", text)
        text = self.WECHAT_RE.sub(" ", text)
        text = self.TS_RE.sub(" ", text)
        text = self.HTML_RE.sub(" ", text)
        return self.MULTI_SPACE_RE.sub(" ", text).strip()

    def compress_repeats(self, text: str) -> str:
        return self.REPEAT_CHAR_RE.sub(lambda m: m.group(1) * 2, text)

    def is_invalid(self, text: str) -> bool:
        low = normalize_ws(text).lower()
        if low in self.INVALID_SET:
            return True
        return not bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text))

    def build_candidate_cuts(self, text: str) -> List[Dict[str, Any]]:
        cuts = []
        seen = set()
        idx = 1
        for match in self.SENTENCE_CUT_RE.finditer(text):
            pos = match.end()
            if 0 < pos < len(text) and pos not in seen:
                cuts.append({"cut_id": f"cut_{idx}", "pos": pos, "trigger": match.group(0)})
                seen.add(pos)
                idx += 1
        return cuts

    def build_default_segments(self, text: str) -> List[Dict[str, Any]]:
        if len(text) <= 30:
            return [{"seg_id": "s1", "start": 0, "end": len(text)}]
        cuts = self.build_candidate_cuts(text)
        if not cuts:
            return [{"seg_id": "s1", "start": 0, "end": len(text)}]
        first = cuts[0]["pos"]
        return [{"seg_id": "s1", "start": 0, "end": first}, {"seg_id": "s2", "start": first, "end": len(text)}]

    def preprocess_one(self, row: RawFeedback) -> CleanFeedback:
        normalized = self.normalize_text(row.raw_text)
        stripped = self.strip_noise(normalized)
        cleaned = self.compress_repeats(stripped)
        is_spam = self.is_invalid(cleaned) or any(p.search(cleaned) for p in self.SPAM_PATTERNS)
        is_template = any(p.search(cleaned) for p in self.TEMPLATE_PATTERNS)
        return CleanFeedback(
            record_id=row.record_id, node_id=row.node_id, source_platform=row.source_platform, publish_time=row.publish_time,
            post_title=row.post_title, post_tag=row.post_tag, raw_text=row.raw_text, normalized_text=normalized,
            cleaned_text=cleaned, source_file=row.source_file, source_url=row.source_url, user_id=row.user_id,
            is_spam=is_spam, is_template=is_template, duplicate_of="",
            context={"source_platform": row.source_platform, "publish_time": row.publish_time, "post_title": row.post_title, "post_tag": row.post_tag},
            candidate_cuts=self.build_candidate_cuts(cleaned), default_segments=self.build_default_segments(cleaned), extra=row.extra,
        )

    def mark_duplicates(self, rows: List[CleanFeedback]) -> List[CleanFeedback]:
        accepted: List[CleanFeedback] = []
        exact_seen: Dict[str, str] = {}
        for row in rows:
            exact_key = stable_hash(row.cleaned_text, prefix="TXT")
            if exact_key in exact_seen:
                row.duplicate_of = exact_seen[exact_key]
                accepted.append(row)
                continue
            if self.enable_near_dup:
                for old in accepted:
                    if SequenceMatcher(None, row.cleaned_text, old.cleaned_text).ratio() >= self.similarity_threshold:
                        row.duplicate_of = old.record_id
                        break
            if not row.duplicate_of:
                exact_seen[exact_key] = row.record_id
            accepted.append(row)
        return rows

    def run(self, rows: List[RawFeedback]) -> List[CleanFeedback]:
        cleaned = [self.preprocess_one(x) for x in rows]
        return self.mark_duplicates(cleaned)


class LabelRetriever:
    def __init__(self, library: LabelLibrary):
        self.library = library

    def retrieve(self, text: str, top_k: int = 8) -> List[LabelDefinition]:
        text = normalize_ws(text)
        scored = []
        for label in self.library.active_labels():
            score = 0
            if label.level_2 and label.level_2 in text:
                score += 2
            for term in label.anchor_terms:
                if term and term in text:
                    score += 3
            for token in [label.level_1, label.level_2]:
                if token and any(ch in text for ch in token[:2]):
                    score += 0
            if score > 0:
                scored.append((score, label))
        scored.sort(key=lambda x: (-x[0], x[1].label_id))
        return [x[1] for x in scored[:top_k]]


class RuleLabelClassifier:
    POS_TERMS = ["好", "不错", "喜欢", "有趣", "优秀", "舒服", "合理", "满意", "流畅", "好看", "很强", "方便", "值得"]
    NEG_TERMS = ["差", "烂", "无聊", "坐牢", "拖沓", "难用", "太少", "装死", "敷衍", "重复", "太弱", "太贵", "空", "烦", "劝退"]

    def __init__(self, library: LabelLibrary, retriever: LabelRetriever):
        self.library = library
        self.retriever = retriever

    def infer_sentiment(self, text: str) -> str:
        pos = any(x in text for x in self.POS_TERMS)
        neg = any(x in text for x in self.NEG_TERMS)
        if pos and not neg:
            return "positive"
        if neg and not pos:
            return "negative"
        return "neutral"

    async def run(self, rows: List[CleanFeedback]) -> List[LabelResult]:
        results = []
        for row in rows:
            if row.is_spam or row.is_template or row.duplicate_of:
                results.append(LabelResult(record_id=row.record_id, hits=[]))
                continue
            query = " ".join([row.cleaned_text, row.post_title, " ".join(row.post_tag)]).strip()
            hits = []
            for label in self.retriever.retrieve(query, top_k=8):
                probe_terms = label.anchor_terms + [label.level_2]
                if any(t and t in query for t in probe_terms):
                    hits.append(LabelHit(label_id=label.label_id, sentiment=self.infer_sentiment(query), confidence=0.56, source="rule"))
            dedup = []
            seen = set()
            for hit in hits:
                if hit.label_id not in seen:
                    dedup.append(hit)
                    seen.add(hit.label_id)
            results.append(LabelResult(record_id=row.record_id, hits=dedup))
        return results


class AsyncLLMClient:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrency)

    async def call_json(self, session: aiohttp.ClientSession, system_prompt: str, user_content: str, max_retries: int = 5) -> Any:
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            "temperature": 0.01,
            "presence_penalty": 0.5,
            "enable_thinking": False,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=300)
        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    async with session.post(self.config.api_base_url, json=payload, headers=headers, timeout=timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            raw_content = data["choices"][0]["message"]["content"] or ""
                            content = strip_think_blocks(raw_content)
                            snippet = extract_first_balanced_json(content)
                            result = robust_json_loads(snippet or content)
                            if result is not None:
                                return result
                            print(f"  ⚠️ [LLM 第{attempt+1}次] JSON 解析失败，准备重试...")
                            continue
                        if resp.status == 429:
                            print(f"  ⏳ 触发限流，避让休眠中... (第 {attempt+1} 次重试)")
                            await asyncio.sleep(3)
                        else:
                            body = await resp.text()
                            print(f"  ⚠️ [LLM] 接口异常 (状态码: {resp.status})，响应: {body[:200]}，准备重试...")
                            await asyncio.sleep(2)
            except Exception as e:
                print(f"  ⚠️ [LLM] 网络异常 ({type(e).__name__}: {e})，准备重试...")
                await asyncio.sleep(3)
        print(f"  ❌ [LLM] 达到最大重试次数 ({max_retries})，返回 None")
        return None


class LLMLabelClassifier:
    SYSTEM_LABELS = {
        "SYS_INVALID": "无效反馈",
        "SYS_LOW_VALUE": "低价值反馈",
        "SYS_UNRESOLVED": "无法解析",
    }

    def __init__(self, library: LabelLibrary, retriever: LabelRetriever, client: AsyncLLMClient, batch_plan: List[int]):
        self.library = library
        self.retriever = retriever
        self.client = client
        self.batch_plan = batch_plan

    def _render_rows(self, rows: List[CleanFeedback]) -> str:
        blocks = []
        for row in rows:
            candidates = self.retriever.retrieve(" ".join([row.cleaned_text, row.post_title, " ".join(row.post_tag)]), top_k=8)
            label_lines = [f"- {x.label_id} | {x.level_1}/{x.level_2} | {x.definition} | 锚点:{'；'.join(x.anchor_terms[:8])}" for x in candidates]
            blocks.append("\n".join([
                f"ID:{row.record_id}",
                f"节点:{row.node_id}",
                f"平台:{row.source_platform}",
                f"标题:{row.post_title}",
                f"标签:{'；'.join(row.post_tag)}",
                f"正文:{row.cleaned_text}",
                "候选标签:",
                *label_lines,
            ]))
        return "\n\n".join(blocks)

    async def _process_chunk(self, session: aiohttp.ClientSession, rows: List[CleanFeedback]) -> Dict[str, Any]:
        system_prompt = """你是一个严格的游戏社区反馈标签判别器。
你只能从候选标签中选择 label_id。
输出纯 JSON，格式：
{
  "RID_1": [{"label_id":"A1","sentiment":"negative"}],
  "RID_2": [{"label_id":"SYS_LOW_VALUE","sentiment":"neutral"}]
}
sentiment 只能是 positive / negative / neutral。
若文本无意义、广告、纯噪声，使用 SYS_INVALID。
若文本只有笼统态度而没有明确对象或原因，使用 SYS_LOW_VALUE。
若无法可靠归类，使用 SYS_UNRESOLVED。
不要输出解释文字。"""
        rendered = self._render_rows(rows)
        print(f"  📡 [LLM Label] 发送 {len(rows)} 条记录, 请求体长度={len(rendered)} chars")
        res = await self.client.call_json(session, system_prompt, rendered, max_retries=5)
        if res is None:
            print(f"  ❌ [LLM Label] call_json 返回 None（API 调用全部失败）")
        elif isinstance(res, dict):
            print(f"  ✅ [LLM Label] 返回 {len(res)} 条记录: {list(res.keys())[:5]}...")
        else:
            print(f"  ⚠️ [LLM Label] 返回类型异常: {type(res).__name__}, 值={str(res)[:200]}")
        return res if isinstance(res, dict) else {}

    async def _handle_chunk(self, session: aiohttp.ClientSession, rows: List[CleanFeedback], batch_size: int) -> Dict[str, Any]:
        res = await self._process_chunk(session, rows)
        expected = max(1, int(len(rows) * 0.85))
        if len(res) >= expected:
            return res
        print(f"  ⚠️ [LLM Label] 覆盖不足 ({len(res)}/{len(rows)}, 需≥{expected}), 尝试拆分...")
        if len(rows) == 1:
            print(f"  ⚠️ [LLM Label] 单条记录 {rows[0].record_id} 仍无法解析，回退 SYS_UNRESOLVED")
            return {rows[0].record_id: [{"label_id": "SYS_UNRESOLVED", "sentiment": "neutral"}]}
        mid = max(1, len(rows) // 2)
        left = await self._handle_chunk(session, rows[:mid], batch_size)
        right = await self._handle_chunk(session, rows[mid:], batch_size)
        merged = {}
        merged.update(left)
        merged.update(right)
        return merged

    def _normalize(self, record_id: str, payload: Any) -> LabelResult:
        items = payload if isinstance(payload, list) else [{"label_id": "SYS_UNRESOLVED", "sentiment": "neutral"}]
        hits = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label_id = normalize_ws(item.get("label_id", ""))
            sentiment = normalize_ws(item.get("sentiment", "neutral")) or "neutral"
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = "neutral"
            if label_id in self.library.label_index or label_id.startswith("SYS_"):
                hits.append(LabelHit(label_id=label_id, sentiment=sentiment, confidence=0.75, source="llm"))
        if not hits:
            hits = [LabelHit(label_id="SYS_UNRESOLVED", sentiment="neutral", confidence=0.0, source="llm")]
        dedup = []
        seen = set()
        for hit in hits:
            if hit.label_id not in seen:
                dedup.append(hit)
                seen.add(hit.label_id)
        return LabelResult(record_id=record_id, hits=dedup, raw_model_payload=payload)

    async def run(self, rows: List[CleanFeedback]) -> List[LabelResult]:
        valid_rows = [x for x in rows if not x.is_spam and not x.is_template and not x.duplicate_of]
        out: Dict[str, LabelResult] = {}
        for x in rows:
            if x.is_spam or x.is_template or x.duplicate_of:
                out[x.record_id] = LabelResult(record_id=x.record_id, hits=[LabelHit(label_id="SYS_INVALID", sentiment="neutral", confidence=1.0, source="system")])
        if valid_rows:
            connector = aiohttp.TCPConnector(limit=self.client.config.max_concurrency)
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=600)) as session:
                payload = {}
                for bs in self.batch_plan:
                    chunks = [valid_rows[i:i+bs] for i in range(0, len(valid_rows), bs)]
                    merged = {}
                    for coro in asyncio.as_completed([self._handle_chunk(session, chunk, bs) for chunk in chunks]):
                        piece = await coro
                        if isinstance(piece, dict):
                            merged.update(piece)
                    payload = merged
                    if len(set(payload.keys()) & {x.record_id for x in valid_rows}) / max(1, len(valid_rows)) >= 0.95:
                        break
            for row in valid_rows:
                out[row.record_id] = self._normalize(row.record_id, payload.get(row.record_id))
        return [out[x.record_id] for x in rows if x.record_id in out]


class RuleBoundaryValidator:
    def __init__(self, library: LabelLibrary):
        self.library = library

    async def run(self, rows: List[CleanFeedback], label_results: List[LabelResult]) -> List[BoundaryResult]:
        result_map = {r.record_id: r for r in label_results}
        out = []
        for row in rows:
            label_ids = [h.label_id for h in result_map.get(row.record_id, LabelResult(record_id=row.record_id)).hits if h.label_id in self.library.label_index]
            groups = {self.library.label_index[x].level_1 for x in label_ids}
            if len(groups) >= 2 and row.candidate_cuts and (len(row.cleaned_text) > 18 or len(label_ids) >= 3):
                out.append(BoundaryResult(record_id=row.record_id, decision="USE", cut_ids=[row.candidate_cuts[0]["cut_id"]], reason="cross_dimension_and_has_cut"))
            else:
                out.append(BoundaryResult(record_id=row.record_id, decision="PASS", cut_ids=[], reason="simple_case"))
        return out


class LLMBoundaryValidator:
    def __init__(self, library: LabelLibrary, client: AsyncLLMClient, batch_plan: List[int]):
        self.library = library
        self.client = client
        self.batch_plan = batch_plan

    def _render_rows(self, rows: List[CleanFeedback], label_map: Dict[str, LabelResult]) -> str:
        blocks = []
        for row in rows:
            hits = []
            for hit in label_map.get(row.record_id, LabelResult(record_id=row.record_id)).hits:
                label = self.library.label_index.get(hit.label_id)
                if label:
                    hits.append(f"{hit.label_id}|{label.level_1}/{label.level_2}|{hit.sentiment}|{label.boundary_hint}")
            blocks.append(json.dumps({"record_id": row.record_id, "text": row.cleaned_text, "candidate_cuts": row.candidate_cuts, "predicted_labels": hits}, ensure_ascii=False))
        return "\n".join(blocks)

    async def _process_chunk(self, session: aiohttp.ClientSession, rows: List[CleanFeedback], label_map: Dict[str, LabelResult]) -> Dict[str, Any]:
        system_prompt = """你是一个边界校验器。
输出纯 JSON：
{
  "RID_1": {"decision":"PASS","cut_ids":[]},
  "RID_2": {"decision":"USE","cut_ids":["cut_1"]}
}
decision 只能是 PASS 或 USE。
若单主题明确则 PASS。
若跨多个语义对象且候选切点可用则 USE。
cut_ids 只能取自 candidate_cuts。
不要输出解释文字。"""
        res = await self.client.call_json(session, system_prompt, self._render_rows(rows, label_map), max_retries=3)
        return res if isinstance(res, dict) else {}

    async def _handle_chunk(self, session: aiohttp.ClientSession, rows: List[CleanFeedback], label_map: Dict[str, LabelResult], batch_size: int) -> Dict[str, Any]:
        res = await self._process_chunk(session, rows, label_map)
        if len(res) >= max(1, int(len(rows) * 0.85)):
            return res
        if len(rows) == 1:
            return {rows[0].record_id: {"decision": "PASS", "cut_ids": []}}
        mid = max(1, len(rows) // 2)
        left = await self._handle_chunk(session, rows[:mid], label_map, batch_size)
        right = await self._handle_chunk(session, rows[mid:], label_map, batch_size)
        merged = {}
        merged.update(left)
        merged.update(right)
        return merged

    async def run(self, rows: List[CleanFeedback], label_results: List[LabelResult]) -> List[BoundaryResult]:
        label_map = {x.record_id: x for x in label_results}
        valid_rows = [x for x in rows if x.candidate_cuts]
        result_map = {x.record_id: BoundaryResult(record_id=x.record_id, decision="PASS", reason="no_candidate_cut") for x in rows}
        if valid_rows:
            connector = aiohttp.TCPConnector(limit=self.client.config.max_concurrency)
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=600)) as session:
                payload = {}
                for bs in self.batch_plan:
                    chunks = [valid_rows[i:i+bs] for i in range(0, len(valid_rows), bs)]
                    merged = {}
                    for coro in asyncio.as_completed([self._handle_chunk(session, chunk, label_map, bs) for chunk in chunks]):
                        piece = await coro
                        if isinstance(piece, dict):
                            merged.update(piece)
                    payload = merged
                    if len(set(payload.keys()) & {x.record_id for x in valid_rows}) / max(1, len(valid_rows)) >= 0.95:
                        break
            for row in valid_rows:
                data = payload.get(row.record_id, {})
                allowed = {x["cut_id"] for x in row.candidate_cuts}
                decision = normalize_ws(data.get("decision", "PASS")) or "PASS"
                if decision not in {"PASS", "USE"}:
                    decision = "PASS"
                cut_ids = [x for x in data.get("cut_ids", []) if x in allowed]
                result_map[row.record_id] = BoundaryResult(record_id=row.record_id, decision=decision, cut_ids=cut_ids, reason="llm", raw_model_payload=data)
        return [result_map[x.record_id] for x in rows]


class ResultBuilder:
    def __init__(self, library: LabelLibrary, review_label_density_threshold: int = 4):
        self.library = library
        self.review_label_density_threshold = review_label_density_threshold

    def build_statistics_rows(self, rows: List[CleanFeedback], label_results: List[LabelResult]) -> List[Dict[str, Any]]:
        row_map = {x.record_id: x for x in rows}
        return [{
            "record_id": result.record_id,
            "node_id": row_map[result.record_id].node_id,
            "source_platform": row_map[result.record_id].source_platform,
            "publish_time": row_map[result.record_id].publish_time,
            "label_hits": [x.label_id for x in result.hits],
            "label_sentiments": {x.label_id: x.sentiment for x in result.hits},
        } for result in label_results]

    def _apply_cuts(self, text: str, boundary: BoundaryResult, candidates: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        if boundary.decision != "USE":
            return [(0, len(text))]
        cut_positions = sorted(x["pos"] for x in candidates if x["cut_id"] in set(boundary.cut_ids))
        if not cut_positions:
            return [(0, len(text))]
        points = [0] + cut_positions + [len(text)]
        return [(points[i], points[i+1]) for i in range(len(points)-1)]

    def _assign(self, segments: List[str], hits: List[LabelHit]) -> List[Dict[str, Any]]:
        out = []
        for idx, seg_text in enumerate(segments, start=1):
            linked_labels, linked_sentiments = [], {}
            for hit in hits:
                label = self.library.label_index.get(hit.label_id)
                if not label:
                    continue
                probe_terms = [label.level_2] + label.anchor_terms
                if any(t and t in seg_text for t in probe_terms):
                    linked_labels.append(hit.label_id)
                    linked_sentiments[hit.label_id] = hit.sentiment
            if not linked_labels and len(hits) == 1:
                linked_labels = [hits[0].label_id]
                linked_sentiments[hits[0].label_id] = hits[0].sentiment
            out.append({"seg_id": f"s{idx}", "text": seg_text, "linked_labels": linked_labels, "linked_sentiments": linked_sentiments})
        return out

    def build_presentation_rows(self, rows: List[CleanFeedback], label_results: List[LabelResult], boundary_results: List[BoundaryResult]) -> List[Dict[str, Any]]:
        row_map = {x.record_id: x for x in rows}
        boundary_map = {x.record_id: x for x in boundary_results}
        out = []
        for result in label_results:
            row = row_map[result.record_id]
            spans = self._apply_cuts(row.cleaned_text, boundary_map[result.record_id], row.candidate_cuts)
            texts = [row.cleaned_text[s:e].strip() for s, e in spans]
            segs = self._assign(texts, result.hits)
            final = []
            for (start, end), seg in zip(spans, segs):
                seg["start"], seg["end"] = start, end
                final.append(seg)
            out.append({"record_id": result.record_id, "node_id": row.node_id, "raw_text": row.raw_text, "segments": final})
        return out

    def build_review_pool(self, rows: List[CleanFeedback], label_results: List[LabelResult], boundary_results: List[BoundaryResult]) -> List[Dict[str, Any]]:
        label_map = {x.record_id: x for x in label_results}
        boundary_map = {x.record_id: x for x in boundary_results}
        review_rows = []
        for row in rows:
            if row.is_spam:
                review_rows.append({"record_id": row.record_id, "failure_stage": "preprocess", "failure_reason": "spam_or_invalid", "review_status": "pending"})
                continue
            if row.is_template:
                review_rows.append({"record_id": row.record_id, "failure_stage": "preprocess", "failure_reason": "template_detected", "review_status": "pending"})
                continue
            if row.duplicate_of:
                review_rows.append({"record_id": row.record_id, "failure_stage": "preprocess", "failure_reason": f"duplicate_of::{row.duplicate_of}", "review_status": "pending"})
                continue
            label_result = label_map.get(row.record_id)
            if label_result and len(label_result.hits) >= self.review_label_density_threshold:
                review_rows.append({"record_id": row.record_id, "failure_stage": "labeling", "failure_reason": "label_density_too_high", "predicted_hits": [x.label_id for x in label_result.hits], "review_status": "pending"})
            boundary_result = boundary_map.get(row.record_id)
            if boundary_result and boundary_result.decision == "USE":
                review_rows.append({"record_id": row.record_id, "failure_stage": "boundary_validation", "failure_reason": "manual_check_recommended", "predicted_hits": [x.label_id for x in label_result.hits] if label_result else [], "boundary_result": asdict(boundary_result), "review_status": "pending"})
        return review_rows

    def build_node_summary(self, statistics_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        node_counter: Dict[str, Counter[str]] = defaultdict(Counter)
        node_sentiments: Dict[str, Counter[str]] = defaultdict(Counter)
        for row in statistics_rows:
            for label_id in row["label_hits"]:
                node_counter[row["node_id"]][label_id] += 1
                node_sentiments[row["node_id"]][row["label_sentiments"][label_id]] += 1
        return [{
            "node_id": node_id,
            "label_frequency": dict(label_freq),
            "sentiment_distribution": dict(node_sentiments[node_id]),
            "total_effective_hits": sum(label_freq.values()),
        } for node_id, label_freq in node_counter.items()]

    # ── 深度分析方法 ────────────────────────────────────────

    @staticmethod
    def _sentiment_score(sentiment: str) -> float:
        """将三值情感映射为数值分: positive=1.0, neutral=0.5, negative=0.0"""
        return {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(sentiment, 0.5)

    @staticmethod
    def _shannon_entropy(counts: Dict[str, int]) -> float:
        """计算香农熵 H(X) = -Σ p·log₂(p)"""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for c in counts.values():
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy

    def build_deep_analytics(self, statistics_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于 statistics_rows 计算深度分析指标:
        1. 声量指数 (Voice Index) - 情感加权
        2. 共识度 (Consensus) - 基于信息熵
        3. TGI 指数 - 按 node_id 分组
        4. 四象限偏好地图数据
        5. 标签共现网络
        """
        label_index = self.library.label_index

        # ── 过滤有效记录（有正式标签命中的） ─────────────────
        effective_rows = [r for r in statistics_rows if any(
            lid in label_index or lid.startswith("SYS_") for lid in r.get("label_hits", [])
        )]

        # ── 收集每个标签的情感列表 ──────────────────────────
        label_sentiments_all: Dict[str, List[str]] = defaultdict(list)  # label_id → [sentiments]
        label_by_node: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))  # node_id → label_id → [sentiments]
        total_effective_hits = 0

        for row in effective_rows:
            sentiments = row.get("label_sentiments", {})
            for lid in row.get("label_hits", []):
                if lid not in label_index:
                    continue
                s = sentiments.get(lid, "neutral")
                label_sentiments_all[lid].append(s)
                label_by_node[row["node_id"]][lid].append(s)
                total_effective_hits += 1

        if total_effective_hits == 0:
            return {"voice_index": [], "consensus": [], "tgi": [], "quadrant": [], "cooccurrence": []}

        # ────────────────────────────────────────────────────
        # 1. 声量指数 & 2. 共识度 & 四象限基础数据
        # ────────────────────────────────────────────────────
        voice_index_list = []
        consensus_list = []
        quadrant_list = []
        max_entropy = math.log2(3)  # 三值分类的最大熵

        for lid, sent_list in label_sentiments_all.items():
            lb = label_index.get(lid)
            if not lb:
                continue
            count = len(sent_list)
            volume_share = count / total_effective_hits

            # 情感均分
            avg_score = sum(self._sentiment_score(s) for s in sent_list) / count

            # 声量指数 = 声量占比 × 情感均分
            vi = round(volume_share * avg_score, 4)
            voice_index_list.append({
                "label_id": lid,
                "level_1": lb.level_1,
                "level_2": lb.level_2,
                "display": f"{lb.level_1}/{lb.level_2}",
                "count": count,
                "volume_share": round(volume_share, 4),
                "avg_sentiment_score": round(avg_score, 4),
                "voice_index": vi,
            })

            # 共识度 = 1 - H / H_max
            sent_counter = Counter(sent_list)
            entropy = self._shannon_entropy(dict(sent_counter))
            consensus = round(1 - (entropy / max_entropy if max_entropy > 0 else 0), 4)
            consensus_list.append({
                "label_id": lid,
                "display": f"{lb.level_1}/{lb.level_2}",
                "count": count,
                "positive_count": sent_counter.get("positive", 0),
                "neutral_count": sent_counter.get("neutral", 0),
                "negative_count": sent_counter.get("negative", 0),
                "entropy": round(entropy, 4),
                "consensus": consensus,
            })

            # 四象限基础数据 (X=声量占比, Y=情感均分)
            quadrant_list.append({
                "label_id": lid,
                "display": f"{lb.level_1}/{lb.level_2}",
                "level_1": lb.level_1,
                "x_volume_share": round(volume_share * 100, 2),  # 百分比形式
                "y_sentiment_score": round(avg_score * 100, 2),  # 百分比形式 (0~100)
                "count": count,
                "consensus": consensus,
            })

        voice_index_list.sort(key=lambda x: -x["voice_index"])
        consensus_list.sort(key=lambda x: x["consensus"])

        # 四象限轴心 = 全局均值
        if quadrant_list:
            avg_x = sum(q["x_volume_share"] for q in quadrant_list) / len(quadrant_list)
            avg_y = sum(q["y_sentiment_score"] for q in quadrant_list) / len(quadrant_list)
        else:
            avg_x, avg_y = 0, 50

        # ────────────────────────────────────────────────────
        # 3. TGI 指数 (按 node_id 分组)
        # ────────────────────────────────────────────────────
        # 全网各维度正面占比
        global_pos_ratio: Dict[str, float] = {}
        for lid, sent_list in label_sentiments_all.items():
            pos = sum(1 for s in sent_list if s == "positive")
            global_pos_ratio[lid] = pos / len(sent_list) if sent_list else 0

        tgi_rows = []
        for node_id, label_dict in label_by_node.items():
            for lid, sent_list in label_dict.items():
                lb = label_index.get(lid)
                if not lb:
                    continue
                node_pos = sum(1 for s in sent_list if s == "positive")
                node_ratio = node_pos / len(sent_list) if sent_list else 0
                global_r = global_pos_ratio.get(lid, 0)
                tgi = round((node_ratio / global_r) * 100, 1) if global_r > 0 else 0.0
                tgi_rows.append({
                    "node_id": node_id,
                    "label_id": lid,
                    "display": f"{lb.level_1}/{lb.level_2}",
                    "node_positive_ratio": round(node_ratio * 100, 1),
                    "global_positive_ratio": round(global_r * 100, 1),
                    "tgi": tgi,
                    "node_count": len(sent_list),
                })
        tgi_rows.sort(key=lambda x: -x["tgi"])

        # ────────────────────────────────────────────────────
        # 5. 标签共现网络
        # ────────────────────────────────────────────────────
        cooccurrence: Counter = Counter()
        for row in effective_rows:
            hits = [lid for lid in row.get("label_hits", []) if lid in label_index]
            for i in range(len(hits)):
                for j in range(i + 1, len(hits)):
                    pair = tuple(sorted([hits[i], hits[j]]))
                    cooccurrence[pair] += 1

        cooccurrence_list = []
        for (lid_a, lid_b), weight in cooccurrence.most_common(50):
            lb_a, lb_b = label_index.get(lid_a), label_index.get(lid_b)
            if lb_a and lb_b:
                cooccurrence_list.append({
                    "source": f"{lb_a.level_1}/{lb_a.level_2}",
                    "source_id": lid_a,
                    "target": f"{lb_b.level_1}/{lb_b.level_2}",
                    "target_id": lid_b,
                    "weight": weight,
                })

        return {
            "voice_index": voice_index_list,
            "consensus": consensus_list,
            "tgi": tgi_rows,
            "quadrant": quadrant_list,
            "quadrant_center": {"x": round(avg_x, 2), "y": round(avg_y, 2)},
            "cooccurrence": cooccurrence_list,
        }


class DashboardBuilder:
    """基于管道产出数据，生成类似舆情监控看板的可视化 HTML 报告。"""

    @staticmethod
    def _esc(value: Any) -> str:
        import html as _html
        return _html.escape(str(value) if value else "")

    @staticmethod
    def _calc_nsr(pos: int, neg: int, total: int) -> float:
        return round((pos - neg) / total * 100, 1) if total else 0.0

    def build(
        self,
        *,
        clean_rows: List[CleanFeedback],
        label_results: List[LabelResult],
        presentation_rows: List[Dict[str, Any]],
        node_summary: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
        library: "LabelLibrary",
        deep_analytics: Optional[Dict[str, Any]] = None,
        node_display_map: Optional[Dict[str, str]] = None,
    ) -> str:
        esc = self._esc
        label_index = library.label_index
        _ndm = node_display_map or {}

        def node_display(nid: str) -> str:
            return _ndm.get(nid, nid)

        # SYS_* 系统标签的可读名称
        SYS_LABEL_NAMES: Dict[str, str] = {
            "SYS_INVALID": "无效反馈",
            "SYS_LOW_VALUE": "低价值反馈",
            "SYS_UNRESOLVED": "无法归类",
        }

        # ── 标签可读名映射 ──────────────────────────────────
        def label_display(lid: str) -> str:
            lb = label_index.get(lid)
            if lb:
                return f"{lb.level_1}/{lb.level_2}"
            return SYS_LABEL_NAMES.get(lid, lid)

        # 判断一个 label_id 是否为"已知标签"（正式标签 或 SYS_* 系统标签）
        def is_known_label(lid: str) -> bool:
            return lid in label_index or lid.startswith("SYS_")

        # ── KPI ─────────────────────────────────────────────
        total_volume = int(diagnostics.get("clean_count", len(clean_rows)))
        spam_count = int(diagnostics.get("spam_count", 0))
        dup_count = int(diagnostics.get("duplicate_count", 0))
        template_count = int(diagnostics.get("template_count", 0))
        node_count = int(diagnostics.get("node_count", 0))

        label_map: Dict[str, LabelResult] = {r.record_id: r for r in label_results}

        pos_count = neg_count = neu_count = 0
        for lr in label_results:
            sentiments = {h.sentiment for h in lr.hits if is_known_label(h.label_id)}
            if "negative" in sentiments:
                neg_count += 1
            elif "positive" in sentiments:
                pos_count += 1
            else:
                neu_count += 1

        effective_total = pos_count + neg_count + neu_count or 1
        pos_pct = round(pos_count / effective_total * 100, 1)
        neg_pct = round(neg_count / effective_total * 100, 1)
        neu_pct = round(neu_count / effective_total * 100, 1)
        nsr = self._calc_nsr(pos_count, neg_count, effective_total)

        # ── 标签频次 Top 10（正/负分别，按节点拆分） ────────
        neg_label_counter: Counter = Counter()
        pos_label_counter: Counter = Counter()
        all_label_counter: Counter = Counter()
        # 按 (label_id, node_id, sentiment) 细分计数
        neg_label_node_counter: Dict[str, Counter] = defaultdict(Counter)  # label_id → {node_display_name: count}
        pos_label_node_counter: Dict[str, Counter] = defaultdict(Counter)
        row_map_for_labels = {r.record_id: r for r in clean_rows}
        for lr in label_results:
            row = row_map_for_labels.get(lr.record_id)
            nid = node_display(row.node_id) if row else "未知节点"
            for h in lr.hits:
                if not is_known_label(h.label_id):
                    continue
                all_label_counter[h.label_id] += 1
                if h.sentiment == "negative":
                    neg_label_counter[h.label_id] += 1
                    neg_label_node_counter[h.label_id][nid] += 1
                elif h.sentiment == "positive":
                    pos_label_counter[h.label_id] += 1
                    pos_label_node_counter[h.label_id][nid] += 1

        # 收集所有出现的节点名称（用于堆叠图的 series）
        all_neg_nodes: set = set()
        all_pos_nodes: set = set()
        neg_top_lids = [lid for lid, _ in neg_label_counter.most_common(10)]
        pos_top_lids = [lid for lid, _ in pos_label_counter.most_common(10)]
        for lid in neg_top_lids:
            all_neg_nodes.update(neg_label_node_counter[lid].keys())
        for lid in pos_top_lids:
            all_pos_nodes.update(pos_label_node_counter[lid].keys())
        neg_node_list = sorted(all_neg_nodes)
        pos_node_list = sorted(all_pos_nodes)

        # 构建堆叠图数据：每个节点是一个 series，数据按标签排列
        neg_tag_labels = [label_display(lid) for lid in neg_top_lids]
        neg_tag_node_series = []
        for nd in neg_node_list:
            neg_tag_node_series.append({
                "节点": nd,
                "数据": [neg_label_node_counter[lid].get(nd, 0) for lid in neg_top_lids],
            })

        pos_tag_labels = [label_display(lid) for lid in pos_top_lids]
        pos_tag_node_series = []
        for nd in pos_node_list:
            pos_tag_node_series.append({
                "节点": nd,
                "数据": [pos_label_node_counter[lid].get(nd, 0) for lid in pos_top_lids],
            })

        # ── 负向 Pareto ─────────────────────────────────────
        neg_pareto = []
        if neg_label_counter:
            neg_total = sum(neg_label_counter.values())
            running = 0
            for lid, cnt in neg_label_counter.most_common():
                running += cnt
                neg_pareto.append({
                    "标签": label_display(lid),
                    "数量": cnt,
                    "累计占比": round(running / neg_total * 100, 1) if neg_total else 0,
                })

        # ── 一级维度分布（治理归因）──────────────────────────
        dim_counter: Counter = Counter()
        for lid, cnt in neg_label_counter.items():
            lb = label_index.get(lid)
            if lb:
                dim_counter[lb.level_1] += cnt
            elif lid.startswith("SYS_"):
                dim_counter[SYS_LABEL_NAMES.get(lid, lid)] += cnt
        governance_rank = [{"分类": k, "数量": v} for k, v in dim_counter.most_common()]

        # ── 平台分布 ────────────────────────────────────────
        platform_counter: Counter = Counter()
        for row in clean_rows:
            platform_counter[row.source_platform] += 1
        platform_rank = [{"平台": k, "数量": v} for k, v in platform_counter.most_common(10)]

        # ── 代表性原声 ──────────────────────────────────────
        row_map = {r.record_id: r for r in clean_rows}

        def pick_voices(sentiment: str, top_n: int = 6) -> List[Dict[str, str]]:
            candidates = []
            for lr in label_results:
                hit_sentiments = {h.sentiment for h in lr.hits if is_known_label(h.label_id)}
                if sentiment not in hit_sentiments:
                    continue
                row = row_map.get(lr.record_id)
                if not row or row.is_spam or row.duplicate_of:
                    continue
                label_names = [label_display(h.label_id) for h in lr.hits if is_known_label(h.label_id)]
                candidates.append({
                    "tag": "、".join(label_names) or "未分类",
                    "platform": row.source_platform,
                    "time": row.publish_time,
                    "text": row.raw_text,
                })
            return candidates[:top_n]

        neg_voices = pick_voices("negative", 6)
        pos_voices = pick_voices("positive", 6)

        def render_voices(items: List[Dict[str, str]], color: str) -> str:
            if not items:
                return '<div class="empty-state">暂无数据</div>'
            blocks = []
            for item in items:
                blocks.append(
                    f'<div class="comment">'
                    f'<div style="margin-bottom:6px;">'
                    f'<span class="tag" style="background:{color};">{esc(item["tag"])}</span>'
                    f'<span class="meta">{esc(item["platform"])} | {esc(item["time"])}</span>'
                    f'</div>'
                    f'<div>{esc(item["text"])}</div>'
                    f'</div>'
                )
            return "".join(blocks)

        neg_voices_html = render_voices(neg_voices, "#c0392b")
        pos_voices_html = render_voices(pos_voices, "#2e8b57")

        # ── 深度分析数据 ────────────────────────────────────
        da = deep_analytics or {}
        voice_index_data = da.get("voice_index", [])
        consensus_data = da.get("consensus", [])
        tgi_data = da.get("tgi", [])
        quadrant_data = da.get("quadrant", [])
        quadrant_center = da.get("quadrant_center", {"x": 0, "y": 50})
        cooccurrence_data = da.get("cooccurrence", [])
        has_deep = bool(voice_index_data or quadrant_data)

        # ── JSON 序列化 ─────────────────────────────────────
        neg_pareto_json = json.dumps(neg_pareto[:10], ensure_ascii=False)
        governance_json = json.dumps(governance_rank, ensure_ascii=False)
        neg_tag_labels_json = json.dumps(neg_tag_labels, ensure_ascii=False)
        neg_tag_series_json = json.dumps(neg_tag_node_series, ensure_ascii=False)
        pos_tag_labels_json = json.dumps(pos_tag_labels, ensure_ascii=False)
        pos_tag_series_json = json.dumps(pos_tag_node_series, ensure_ascii=False)
        platform_json = json.dumps(platform_rank, ensure_ascii=False)
        voice_index_json = json.dumps(voice_index_data[:20], ensure_ascii=False)
        consensus_json = json.dumps(consensus_data[:20], ensure_ascii=False)
        tgi_json = json.dumps(tgi_data[:30], ensure_ascii=False)
        quadrant_json = json.dumps(quadrant_data, ensure_ascii=False)
        quadrant_center_json = json.dumps(quadrant_center, ensure_ascii=False)
        cooccurrence_json = json.dumps(cooccurrence_data[:30], ensure_ascii=False)

        # ── 时间 ────────────────────────────────────────────
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # ── HTML ────────────────────────────────────────────
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>社区反馈分析看板</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:20px;background:#f5f7fb;color:#1f2d3d;font-family:"Microsoft YaHei","PingFang SC",sans-serif}}
.container{{max-width:1440px;margin:0 auto}}
.header{{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin-bottom:20px;padding-bottom:14px;border-bottom:3px solid #1f2d3d}}
.header h1{{margin:0;font-size:28px;line-height:1.2}}
.sub{{color:#68707c;font-size:13px;margin-top:6px}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:18px}}
.grid-2{{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;margin-bottom:18px}}
.grid-2e{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:18px}}
.card{{background:#fff;border-radius:12px;box-shadow:0 4px 18px rgba(31,45,61,.08);padding:18px 18px 16px}}
.kpi-title{{font-size:13px;color:#6b7280;margin-bottom:8px}}
.kpi-value{{font-size:34px;font-weight:700;line-height:1.1}}
.kpi-sub{{margin-top:8px;color:#6b7280;font-size:12px}}
.green{{color:#2e8b57}}.red{{color:#c0392b}}.gray{{color:#7f8c8d}}
.bar{{display:flex;height:18px;border-radius:999px;overflow:hidden;background:#eef1f5;margin-top:10px}}
.bar>div{{height:100%}}
.chart-box{{height:380px}}.chart-box.tall{{height:460px}}
.section-title{{margin:0 0 12px 0;font-size:16px;font-weight:700}}
.comment{{background:#fafbfc;border-left:4px solid #d0d7de;padding:12px;margin-bottom:10px;border-radius:6px;font-size:13px;line-height:1.65}}
.tag{{display:inline-block;font-size:11px;color:#fff;padding:2px 8px;border-radius:999px;margin-right:8px;vertical-align:middle}}
.meta{{font-size:12px;color:#6b7280}}
.empty-state{{color:#9aa3af;font-size:13px;padding:24px 0}}
.quality-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:10px}}
.quality-item{{text-align:center;padding:10px;background:#f9fafb;border-radius:8px}}
.quality-item .num{{font-size:22px;font-weight:700;color:#1f2d3d}}
.quality-item .lbl{{font-size:12px;color:#6b7280;margin-top:4px}}
@media(max-width:1200px){{.grid-4,.grid-3,.grid-2,.grid-2e{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="container">

<!-- ===== HEADER ===== -->
<div class="header">
<div>
  <h1>社区反馈分析看板</h1>
  <div class="sub">生成时间：{gen_time} | 模式：{'LLM' if diagnostics.get('llm_enabled') else '规则'} | 节点数：{node_count}</div>
</div>
<div class="sub">Community Feedback Pipeline 自动生成</div>
</div>

<!-- ===== KPI 卡片 ===== -->
<div class="grid-4">
<div class="card">
  <div class="kpi-title">总声量</div>
  <div class="kpi-value">{total_volume}</div>
  <div class="kpi-sub">经清洗后有效评论条数</div>
</div>
<div class="card">
  <div class="kpi-title">净情感值 NSR</div>
  <div class="kpi-value {'red' if nsr<0 else 'green'}">{nsr}</div>
  <div class="kpi-sub">(正面 − 负面) / 总量 × 100</div>
</div>
<div class="card">
  <div class="kpi-title">情感结构</div>
  <div class="bar">
    <div style="width:{pos_pct}%;background:#2e8b57;" title="正面 {pos_pct}%"></div>
    <div style="width:{neu_pct}%;background:#aab2bd;" title="中性 {neu_pct}%"></div>
    <div style="width:{neg_pct}%;background:#c0392b;" title="负面 {neg_pct}%"></div>
  </div>
  <div class="kpi-sub">
    <span class="green">正面 {pos_count} / {pos_pct}%</span> |
    <span class="gray">中性 {neu_count} / {neu_pct}%</span> |
    <span class="red">负面 {neg_count} / {neg_pct}%</span>
  </div>
</div>
<div class="card">
  <div class="kpi-title">舆情判断</div>
  <div class="kpi-value {'red' if neg_count>pos_count else 'green'}">{"偏负向" if neg_count>pos_count else "偏正向"}</div>
  <div class="kpi-sub">依据正负向声量对比自动判定</div>
</div>
</div>

<!-- ===== 数据质量概览 ===== -->
<div class="card" style="margin-bottom:18px">
<h3 class="section-title">数据质量概览</h3>
<div class="quality-grid">
  <div class="quality-item"><div class="num">{total_volume}</div><div class="lbl">清洗后总量</div></div>
  <div class="quality-item"><div class="num red">{spam_count}</div><div class="lbl">垃圾/无效</div></div>
  <div class="quality-item"><div class="num gray">{dup_count}</div><div class="lbl">近似重复</div></div>
  <div class="quality-item"><div class="num gray">{template_count}</div><div class="lbl">模板内容</div></div>
</div>
</div>

<!-- ===== 负向 Pareto + 治理归因 ===== -->
<div class="grid-2">
<div class="card">
  <h3 class="section-title">负向问题集中度（Pareto）</h3>
  <div id="paretoChart" class="chart-box"></div>
</div>
<div class="card">
  <h3 class="section-title">负向问题治理归因（一级维度）</h3>
  <div id="governanceChart" class="chart-box"></div>
</div>
</div>

<!-- ===== 标签 Top10（按节点堆叠） ===== -->
<div class="grid-2e">
<div class="card">
  <h3 class="section-title red">高频负向标签 Top 10（按节点分布）</h3>
  <div class="sub" style="margin:-8px 0 10px;">纵轴 = 标签名 · 颜色 = 观测节点 · 堆叠长度 = 该标签在对应节点的出现频次</div>
  <div id="negTagChart" class="chart-box tall"></div>
</div>
<div class="card">
  <h3 class="section-title green">高频正向标签 Top 10（按节点分布）</h3>
  <div class="sub" style="margin:-8px 0 10px;">纵轴 = 标签名 · 颜色 = 观测节点 · 堆叠长度 = 该标签在对应节点的出现频次</div>
  <div id="posTagChart" class="chart-box tall"></div>
</div>
</div>

<!-- ===== 平台分布 ===== -->
<div class="card" style="margin-bottom:18px">
  <h3 class="section-title">来源平台分布</h3>
  <div id="platformChart" class="chart-box"></div>
</div>

<!-- ===== 代表性原声 ===== -->
<div class="grid-2e">
<div class="card">
  <h3 class="section-title red">负向代表性原声</h3>
  {neg_voices_html}
</div>
<div class="card">
  <h3 class="section-title green">正向代表性原声</h3>
  {pos_voices_html}
</div>
</div>

{'<!-- ===== 深度分析模块 ===== -->' if has_deep else ''}
{'<div class="header" style="margin-top:28px;"><div><h1>深度分析模块</h1><div class="sub">声量指数 · 共识度 · TGI指数 · 四象限偏好地图 · 标签共现网络</div></div></div>' if has_deep else ''}

{'<div class="grid-2e">' if has_deep else ''}
{'<div class="card"><h3 class="section-title">声量指数排行（情感加权）</h3><div class="sub" style="margin:-8px 0 10px;">声量指数 = (维度评论占比) × 情感均分 | 综合讨论热度与情感倾向</div><div id="voiceIndexChart" class="chart-box tall"></div></div>' if has_deep else ''}
{'<div class="card"><h3 class="section-title">共识度分析（熵值法）</h3><div class="sub" style="margin:-8px 0 10px;">共识度 = 1 − 信息熵/最大熵 | 越接近1，社群观点越一致</div><div id="consensusChart" class="chart-box tall"></div></div>' if has_deep else ''}
{'</div>' if has_deep else ''}

{'<div class="card" style="margin-bottom:18px"><h3 class="section-title">四象限偏好地图</h3><div class="sub" style="margin:-8px 0 10px;">X轴 = 关注度(声量占比%) · Y轴 = 满意度(情感均分%) · 气泡大小 = 评论数 · 虚线 = 全局均值</div><div id="quadrantChart" style="height:560px"></div></div>' if has_deep else ''}

{'<div class="grid-2e">' if has_deep else ''}
{'<div class="card"><h3 class="section-title">TGI 偏好指数（社群纵切）</h3><div class="sub" style="margin:-8px 0 10px;">TGI = (社群正面占比 / 全网正面占比) × 100 | >100 偏好高于均值, <100 低于均值</div><div id="tgiChart" style="height:480px"></div></div>' if has_deep else ''}
{'<div class="card"><h3 class="section-title">标签共现网络</h3><div class="sub" style="margin:-8px 0 10px;">连线粗细 = 共现频次 | 揭示玩家同时讨论哪些维度</div><div id="cooccurrenceChart" style="height:480px"></div></div>' if has_deep else ''}
{'</div>' if has_deep else ''}

</div><!-- /container -->

<script>
const negPareto={neg_pareto_json};
const govRank={governance_json};
const negTagLabels={neg_tag_labels_json};
const negTagSeries={neg_tag_series_json};
const posTagLabels={pos_tag_labels_json};
const posTagSeries={pos_tag_series_json};
const platformData={platform_json};

/* Pareto */
const pc=echarts.init(document.getElementById('paretoChart'));
pc.setOption({{
  tooltip:{{trigger:'axis'}},legend:{{data:['负向数量','累计占比']}},
  grid:{{left:'8%',right:'8%',bottom:'14%',containLabel:true}},
  xAxis:{{type:'category',data:negPareto.map(x=>x['标签']),axisLabel:{{interval:0,rotate:35}}}},
  yAxis:[{{type:'value',name:'数量'}},{{type:'value',name:'累计占比',min:0,max:100,axisLabel:{{formatter:'{{value}}%'}}}}],
  series:[
    {{name:'负向数量',type:'bar',data:negPareto.map(x=>x['数量']),itemStyle:{{color:'#c0392b'}}}},
    {{name:'累计占比',type:'line',yAxisIndex:1,smooth:true,data:negPareto.map(x=>x['累计占比']),itemStyle:{{color:'#1f2d3d'}},label:{{show:true,formatter:'{{c}}%'}}}}
  ]
}});

/* 治理归因 */
const gc=echarts.init(document.getElementById('governanceChart'));
gc.setOption({{
  tooltip:{{trigger:'item'}},
  series:[{{type:'pie',radius:['35%','68%'],center:['50%','52%'],
    label:{{formatter:'{{b}}\\n{{d}}%'}},
    data:govRank.map(x=>({{name:x['分类'],value:x['数量']}}))
  }}]
}});

/* 负向标签（按节点堆叠） */
const NEG_NODE_COLORS=['#c0392b','#e74c3c','#f39c12','#e67e22','#8e44ad','#2c3e50','#16a085','#d35400','#7f8c8d','#2980b9','#1abc9c','#9b59b6'];
const nc=echarts.init(document.getElementById('negTagChart'));
nc.setOption({{
  tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}}}},
  legend:{{data:negTagSeries.map(x=>x['节点']),top:0,type:'scroll',textStyle:{{fontSize:11}}}},
  grid:{{left:'40%',right:'8%',top:'30',bottom:'3%',containLabel:false}},
  xAxis:{{type:'value',name:'频次'}},
  yAxis:{{type:'category',data:negTagLabels.slice().reverse()}},
  series:negTagSeries.map(function(s,i){{
    return {{
      name:s['节点'],type:'bar',stack:'neg_total',
      data:s['数据'].slice().reverse(),
      label:{{show:false}},
      itemStyle:{{color:NEG_NODE_COLORS[i%NEG_NODE_COLORS.length]}},
      emphasis:{{focus:'series'}}
    }};
  }})
}});

/* 正向标签（按节点堆叠） */
const POS_NODE_COLORS=['#2e8b57','#27ae60','#3498db','#2980b9','#1abc9c','#16a085','#f1c40f','#e67e22','#9b59b6','#34495e','#95a5a6','#d35400'];
const ptc=echarts.init(document.getElementById('posTagChart'));
ptc.setOption({{
  tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}}}},
  legend:{{data:posTagSeries.map(x=>x['节点']),top:0,type:'scroll',textStyle:{{fontSize:11}}}},
  grid:{{left:'40%',right:'8%',top:'30',bottom:'3%',containLabel:false}},
  xAxis:{{type:'value',name:'频次'}},
  yAxis:{{type:'category',data:posTagLabels.slice().reverse()}},
  series:posTagSeries.map(function(s,i){{
    return {{
      name:s['节点'],type:'bar',stack:'pos_total',
      data:s['数据'].slice().reverse(),
      label:{{show:false}},
      itemStyle:{{color:POS_NODE_COLORS[i%POS_NODE_COLORS.length]}},
      emphasis:{{focus:'series'}}
    }};
  }})
}});

/* 平台分布 */
const plc=echarts.init(document.getElementById('platformChart'));
plc.setOption({{
  tooltip:{{trigger:'item'}},
  series:[{{type:'pie',radius:['35%','68%'],center:['50%','52%'],
    label:{{formatter:'{{b}}\\n{{c}}条 ({{d}}%)'}},
    data:platformData.map(x=>({{name:x['平台'],value:x['数量']}}))
  }}]
}});

window.addEventListener('resize',function(){{pc.resize();gc.resize();nc.resize();ptc.resize();plc.resize()}});

/* ===== 深度分析图表 ===== */
const voiceData={voice_index_json};
const consensusData={consensus_json};
const tgiData={tgi_json};
const quadrantData={quadrant_json};
const quadrantCenter={quadrant_center_json};
const cooccurrenceData={cooccurrence_json};

const deepCharts=[];

if(voiceData.length>0 && document.getElementById('voiceIndexChart')){{
  const vic=echarts.init(document.getElementById('voiceIndexChart'));
  deepCharts.push(vic);
  const viLabels=voiceData.map(x=>x.display).reverse();
  const viValues=voiceData.map(x=>x.voice_index).reverse();
  const viCounts=voiceData.map(x=>x.count).reverse();
  const viScores=voiceData.map(x=>x.avg_sentiment_score).reverse();
  vic.setOption({{
    tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}},
      formatter:function(p){{
        const i=p[0].dataIndex;
        return '<b>'+viLabels[i]+'</b><br/>声量指数: '+viValues[i]+'<br/>评论数: '+viCounts[i]+'<br/>情感均分: '+viScores[i];
      }}
    }},
    grid:{{left:'40%',right:'10%',top:'3%',bottom:'3%',containLabel:false}},
    xAxis:{{type:'value',name:'声量指数'}},
    yAxis:{{type:'category',data:viLabels}},
    series:[{{
      type:'bar',data:viValues.map((v,i)=>({{
        value:v,
        itemStyle:{{color:viScores[i]>=0.6?'#2e8b57':viScores[i]>=0.4?'#aab2bd':'#c0392b'}}
      }})),
      label:{{show:true,position:'right',formatter:function(p){{return p.value.toFixed(4)}}}}
    }}]
  }});
}}

if(consensusData.length>0 && document.getElementById('consensusChart')){{
  const csc=echarts.init(document.getElementById('consensusChart'));
  deepCharts.push(csc);
  const csLabels=consensusData.map(x=>x.display);
  const csValues=consensusData.map(x=>x.consensus);
  const csPos=consensusData.map(x=>x.positive_count);
  const csNeu=consensusData.map(x=>x.neutral_count);
  const csNeg=consensusData.map(x=>x.negative_count);
  csc.setOption({{
    tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}},
      formatter:function(p){{
        const i=p[0].dataIndex;
        return '<b>'+csLabels[i]+'</b><br/>共识度: '+csValues[i]+'<br/>正面:'+csPos[i]+' 中性:'+csNeu[i]+' 负面:'+csNeg[i];
      }}
    }},
    grid:{{left:'40%',right:'10%',top:'3%',bottom:'3%',containLabel:false}},
    xAxis:{{type:'value',name:'共识度',min:0,max:1}},
    yAxis:{{type:'category',data:csLabels}},
    series:[{{
      type:'bar',data:csValues.map(v=>({{
        value:v,
        itemStyle:{{color:v>=0.7?'#2e8b57':v>=0.4?'#e67e22':'#c0392b'}}
      }})),
      label:{{show:true,position:'right',formatter:function(p){{return p.value.toFixed(2)}}}}
    }}],
    visualMap:{{show:false,min:0,max:1,inRange:{{color:['#c0392b','#e67e22','#2e8b57']}}}}
  }});
}}

if(quadrantData.length>0 && document.getElementById('quadrantChart')){{
  const qc=echarts.init(document.getElementById('quadrantChart'));
  deepCharts.push(qc);
  const L1_COLORS={{'剧情叙事':'#e74c3c','角色机制':'#3498db','玩法机制':'#2ecc71','地图探索':'#f39c12','养成系统':'#9b59b6','运营活动':'#1abc9c','社区环境':'#e67e22','运营沟通':'#95a5a6'}};
  const l1Set=[...new Set(quadrantData.map(x=>x.level_1))];
  const seriesMap={{}};
  quadrantData.forEach(function(d){{
    if(!seriesMap[d.level_1])seriesMap[d.level_1]=[];
    seriesMap[d.level_1].push([d.x_volume_share,d.y_sentiment_score,d.count,d.display,d.consensus]);
  }});
  const qSeries=l1Set.map(function(l1){{
    return {{
      name:l1,type:'scatter',
      symbolSize:function(d){{return Math.max(12,Math.min(50,d[2]*5))}},
      itemStyle:{{color:L1_COLORS[l1]||'#7f8c8d',opacity:0.8}},
      data:seriesMap[l1]||[],
      label:{{show:true,formatter:function(p){{return p.value[3].split('/')[1]}},position:'right',fontSize:11}},
      emphasis:{{focus:'series',label:{{show:true,fontSize:13,fontWeight:'bold'}}}}
    }};
  }});
  qSeries.push({{
    type:'line',markLine:{{silent:true,symbol:'none',lineStyle:{{type:'dashed',color:'#95a5a6',width:1.5}},
      data:[
        {{xAxis:quadrantCenter.x,label:{{formatter:'关注度均值',position:'end'}}}},
        {{yAxis:quadrantCenter.y,label:{{formatter:'满意度均值',position:'end'}}}}
      ]
    }}
  }});
  qc.setOption({{
    tooltip:{{trigger:'item',formatter:function(p){{
      if(!p.value||p.value.length<5)return '';
      return '<b>'+p.value[3]+'</b><br/>关注度: '+p.value[0]+'%<br/>满意度: '+p.value[1]+'%<br/>评论数: '+p.value[2]+'<br/>共识度: '+p.value[4];
    }}}},
    legend:{{data:l1Set,top:0}},
    grid:{{left:'8%',right:'8%',top:'50',bottom:'10%',containLabel:true}},
    xAxis:{{type:'value',name:'关注度 (声量占比%)',nameLocation:'center',nameGap:30,splitLine:{{lineStyle:{{type:'dashed',opacity:0.3}}}}}},
    yAxis:{{type:'value',name:'满意度 (情感均分%)',min:0,max:100,nameLocation:'center',nameGap:40,splitLine:{{lineStyle:{{type:'dashed',opacity:0.3}}}}}},
    series:qSeries,
    graphic:[
      {{type:'text',left:'12%',top:'8%',style:{{text:'🟢 机会区\\n(低关注·高满意)',fill:'#2e8b57',fontSize:12,fontWeight:'bold'}}}},
      {{type:'text',right:'12%',top:'8%',style:{{text:'⭐ 优势区\\n(高关注·高满意)',fill:'#3498db',fontSize:12,fontWeight:'bold'}}}},
      {{type:'text',left:'12%',bottom:'14%',style:{{text:'⚪ 次要区\\n(低关注·低满意)',fill:'#95a5a6',fontSize:12,fontWeight:'bold'}}}},
      {{type:'text',right:'12%',bottom:'14%',style:{{text:'🔴 改进区\\n(高关注·低满意)',fill:'#c0392b',fontSize:12,fontWeight:'bold'}}}}
    ]
  }});
}}

if(tgiData.length>0 && document.getElementById('tgiChart')){{
  const tc=echarts.init(document.getElementById('tgiChart'));
  deepCharts.push(tc);
  const tgiLabels=tgiData.map(x=>x.node_id.replace('node_','')+' | '+x.display);
  const tgiValues=tgiData.map(x=>x.tgi);
  tc.setOption({{
    tooltip:{{trigger:'axis',axisPointer:{{type:'shadow'}},
      formatter:function(p){{
        const d=tgiData[tgiData.length-1-p[0].dataIndex];
        return '<b>'+d.node_id+'</b><br/>'+d.display+'<br/>TGI: '+d.tgi+'<br/>社群正面率: '+d.node_positive_ratio+'%<br/>全网正面率: '+d.global_positive_ratio+'%<br/>样本数: '+d.node_count;
      }}
    }},
    grid:{{left:'45%',right:'10%',top:'3%',bottom:'3%',containLabel:false}},
    xAxis:{{type:'value',name:'TGI',splitLine:{{lineStyle:{{type:'dashed'}}}}}},
    yAxis:{{type:'category',data:tgiLabels.slice().reverse()}},
    series:[{{
      type:'bar',
      data:tgiValues.slice().reverse().map(v=>({{
        value:v,
        itemStyle:{{color:v>=120?'#2e8b57':v>=80?'#aab2bd':'#c0392b'}}
      }})),
      label:{{show:true,position:'right',formatter:function(p){{return p.value}}}},
      markLine:{{silent:true,symbol:'none',lineStyle:{{type:'dashed',color:'#e74c3c',width:1.5}},
        data:[{{xAxis:100,label:{{formatter:'基准线 100',position:'end'}}}}]
      }}
    }}]
  }});
}}

if(cooccurrenceData.length>0 && document.getElementById('cooccurrenceChart')){{
  const cc=echarts.init(document.getElementById('cooccurrenceChart'));
  deepCharts.push(cc);
  const nodeSet=new Set();
  cooccurrenceData.forEach(function(e){{nodeSet.add(e.source);nodeSet.add(e.target)}});
  const nodes=[...nodeSet].map(function(n){{return {{name:n,symbolSize:20+cooccurrenceData.filter(e=>e.source===n||e.target===n).reduce((s,e)=>s+e.weight,0)*3}}}});
  const links=cooccurrenceData.map(function(e){{return {{source:e.source,target:e.target,value:e.weight,lineStyle:{{width:Math.max(1,e.weight*2),opacity:0.6}}}}}});
  cc.setOption({{
    tooltip:{{trigger:'item',formatter:function(p){{
      if(p.dataType==='edge')return p.data.source+' ↔ '+p.data.target+'<br/>共现次数: '+p.data.value;
      return p.name;
    }}}},
    series:[{{
      type:'graph',layout:'force',
      data:nodes,links:links,
      roam:true,draggable:true,
      label:{{show:true,fontSize:11}},
      force:{{repulsion:220,gravity:0.1,edgeLength:[60,200]}},
      emphasis:{{focus:'adjacency',lineStyle:{{width:4}}}},
      lineStyle:{{color:'#aaa',curveness:0.15}}
    }}]
  }});
}}

window.addEventListener('resize',function(){{deepCharts.forEach(function(c){{c.resize()}});}});
</script>
</body>
</html>"""
        return html_content


class ReportWriter:
    def __init__(self):
        self.dashboard_builder = DashboardBuilder()

    def write_all(
        self, *, job: JobContext, rows: List[RawFeedback], clean_rows: List[CleanFeedback], label_results: List[LabelResult],
        boundary_results: List[BoundaryResult], statistics_rows: List[Dict[str, Any]], presentation_rows: List[Dict[str, Any]],
        review_pool: List[Dict[str, Any]], node_summary: List[Dict[str, Any]], diagnostics: Dict[str, Any],
        library: Optional["LabelLibrary"] = None,
        deep_analytics: Optional[Dict[str, Any]] = None,
        node_display_map: Optional[Dict[str, str]] = None,
    ) -> None:
        safe_json_dump(job.assets_dir / "raw_feedback.json", [asdict(x) for x in rows])
        safe_json_dump(job.assets_dir / "preprocess_output.json", [asdict(x) for x in clean_rows])
        safe_json_dump(job.assets_dir / "label_results.json", [{"record_id": x.record_id, "hits": [asdict(h) for h in x.hits]} for x in label_results])
        safe_json_dump(job.assets_dir / "boundary_results.json", [asdict(x) for x in boundary_results])
        safe_json_dump(job.assets_dir / "statistics_rows.json", statistics_rows)
        safe_json_dump(job.assets_dir / "presentation_rows.json", presentation_rows)
        safe_json_dump(job.assets_dir / "review_pool.json", review_pool)
        safe_json_dump(job.assets_dir / "node_summary.json", node_summary)
        safe_json_dump(job.assets_dir / "diagnostics.json", diagnostics)
        if deep_analytics:
            safe_json_dump(job.assets_dir / "deep_analytics.json", deep_analytics)

        pd.DataFrame(statistics_rows).to_csv(job.reports_dir / "statistics_rows.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(review_pool).to_csv(job.reports_dir / "review_pool.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(node_summary).to_csv(job.reports_dir / "node_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "record_id": row.record_id,
            "node_id": row.node_id,
            "source_platform": row.source_platform,
            "publish_time": row.publish_time,
            "post_title": row.post_title,
            "post_tag": "；".join(row.post_tag),
            "raw_text": row.raw_text,
            "cleaned_text": row.cleaned_text,
            "duplicate_of": row.duplicate_of,
            "is_spam": row.is_spam,
            "is_template": row.is_template,
        } for row in clean_rows]).to_csv(job.reports_dir / "preprocess_overview.csv", index=False, encoding="utf-8-sig")

        # ── 自动生成 HTML 可视化看板 ─────────────────────────
        if library is not None:
            try:
                html_content = self.dashboard_builder.build(
                    clean_rows=clean_rows,
                    label_results=label_results,
                    presentation_rows=presentation_rows,
                    node_summary=node_summary,
                    diagnostics=diagnostics,
                    library=library,
                    deep_analytics=deep_analytics,
                    node_display_map=node_display_map,
                )
                dashboard_path = job.reports_dir / "dashboard.html"
                with open(dashboard_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"  ✅ 可视化看板已生成：{dashboard_path}")
            except Exception as e:
                print(f"  ⚠️ 看板生成失败（不影响主流程）：{e}")


class CommunityFeedbackPipeline:
    def __init__(self, runtime_paths: RuntimePaths, config: PipelineConfig):
        self.runtime_paths = runtime_paths
        self.config = config
        self.workspace_manager = WorkspaceManager(runtime_paths)
        self.meta_store = JobMetaStore()
        self.loader = FeedbackLoader()
        self.preprocessor = FeedbackPreprocessor(config.similarity_threshold, config.enable_near_dup)
        self.report_writer = ReportWriter()

    def _build_engines(self, library: LabelLibrary):
        retriever = LabelRetriever(library)
        if self.config.llm_enabled:
            client = AsyncLLMClient(self.config)
            label_classifier = LLMLabelClassifier(library, retriever, client, self.config.label_batch_plan)
            boundary_validator = LLMBoundaryValidator(library, client, self.config.boundary_batch_plan)
        else:
            label_classifier = RuleLabelClassifier(library, retriever)
            boundary_validator = RuleBoundaryValidator(library)
        return label_classifier, boundary_validator, ResultBuilder(library, self.config.review_label_density_threshold)

    async def run(self) -> JobContext:
        self.runtime_paths.ensure()
        self.config.validate()
        library = LabelLibrary.from_path(self.config.label_library_path)
        label_classifier, boundary_validator, result_builder = self._build_engines(library)

        job = self.workspace_manager.prepare_single_job()
        self.meta_store.update(job.meta_path, status="running", label_library_path=str(self.config.label_library_path))
        diagnostics = {
            "llm_enabled": self.config.llm_enabled,
            "model_name": self.config.model_name if self.config.llm_enabled else "rule_mode",
            "label_batch_plan": self.config.label_batch_plan,
            "boundary_batch_plan": self.config.boundary_batch_plan,
        }
        try:
            self.meta_store.update_stage(job.meta_path, "ingest", "running")
            raw_rows = self.loader.load_dir(job.input_dir)
            self.meta_store.update_stage(job.meta_path, "ingest", "done")
            diagnostics["raw_count"] = len(raw_rows)

            self.meta_store.update_stage(job.meta_path, "preprocess", "running")
            clean_rows = self.preprocessor.run(raw_rows)
            self.meta_store.update_stage(job.meta_path, "preprocess", "done")
            diagnostics["clean_count"] = len(clean_rows)
            diagnostics["spam_count"] = sum(1 for x in clean_rows if x.is_spam)
            diagnostics["template_count"] = sum(1 for x in clean_rows if x.is_template)
            diagnostics["duplicate_count"] = sum(1 for x in clean_rows if x.duplicate_of)

            self.meta_store.update_stage(job.meta_path, "aggregate", "done")
            diagnostics["node_count"] = len({x.node_id for x in clean_rows})

            self.meta_store.update_stage(job.meta_path, "labeling", "running")
            label_results = await label_classifier.run(clean_rows)
            self.meta_store.update_stage(job.meta_path, "labeling", "done")

            self.meta_store.update_stage(job.meta_path, "boundary_validation", "running")
            boundary_results = await boundary_validator.run(clean_rows, label_results)
            self.meta_store.update_stage(job.meta_path, "boundary_validation", "done")

            self.meta_store.update_stage(job.meta_path, "reporting", "running")
            statistics_rows = result_builder.build_statistics_rows(clean_rows, label_results)
            presentation_rows = result_builder.build_presentation_rows(clean_rows, label_results, boundary_results)
            review_pool = result_builder.build_review_pool(clean_rows, label_results, boundary_results)
            node_summary = result_builder.build_node_summary(statistics_rows)
            deep_analytics = result_builder.build_deep_analytics(statistics_rows)

            # ── 加载 .mm 文件构建节点名映射 ─────────────────
            mm_path = self.runtime_paths.app_root / "星铁中国大陆网络社区生态地图.mm"
            all_node_ids = {x.node_id for x in clean_rows}
            node_display_map = MindMapNodeMapper.build_runtime_mapping(mm_path, all_node_ids)
            if node_display_map:
                print(f"  ✅ 从 .mm 文件加载了 {len(node_display_map)} 个节点名映射")

            self.report_writer.write_all(
                job=job, rows=raw_rows, clean_rows=clean_rows, label_results=label_results, boundary_results=boundary_results,
                statistics_rows=statistics_rows, presentation_rows=presentation_rows, review_pool=review_pool, node_summary=node_summary,
                diagnostics=diagnostics, library=library, deep_analytics=deep_analytics, node_display_map=node_display_map,
            )
            self.meta_store.update_stage(job.meta_path, "reporting", "done")
            self.meta_store.update(job.meta_path, status="done")
            return job
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            self.meta_store.update(job.meta_path, status="failed", error=err)
            raise


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Community Feedback Pipeline")
    parser.add_argument("--label-library", default="", help="标签库 JSON 路径")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 模式")
    parser.add_argument("--model", default=os.environ.get("CF_MODEL_NAME", "qwen3.5-plus"))
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    return parser.parse_args(argv)


async def main() -> None:
    args = parse_args(sys.argv[1:])
    runtime_paths = RuntimePaths.from_env()
    default_label_library = runtime_paths.app_root / "label_libraries" / "star_rail_feedback_labels.v3.0.0.json"
    config = PipelineConfig(
        label_library_path=Path(args.label_library) if args.label_library else default_label_library,
        llm_enabled=bool(args.llm or env_bool("CF_LLM_ENABLED", False)),
        api_key=args.api_key or os.environ.get("DASHSCOPE_API_KEY", "").strip(),
        model_name=args.model,
        max_concurrency=safe_int(os.environ.get("CF_MAX_CONCURRENCY", "5"), 5),
        similarity_threshold=safe_float(os.environ.get("CF_DUP_THRESHOLD", "0.92"), 0.92),
        enable_near_dup=env_bool("CF_ENABLE_NEAR_DUP", True),
        review_label_density_threshold=safe_int(os.environ.get("CF_REVIEW_LABEL_DENSITY", "4"), 4),
    )
    pipeline = CommunityFeedbackPipeline(runtime_paths, config)
    job = await pipeline.run()
    print("\n处理完成")
    print(f"Job 目录: {job.job_dir}")
    print(f"机器资产: {job.assets_dir}")
    print(f"人工报表: {job.reports_dir}")


if __name__ == "__main__":
    asyncio.run(main())
