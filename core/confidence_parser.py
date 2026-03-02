# core/confidence_parser.py
# -*- coding: utf-8 -*-

"""
工业版置信度/审查结果解析器（完整版）

用途：
1) 解析 Gamma / 审查器返回中的 confidence / score / verdict / issues / minimal_fix
2) 兼容 JSON、伪 JSON、键值对、自然语言格式
3) 给 orchestrator 提供统一接口，用于“是否重试 / 是否强制终审”

注意：
- 本文件提供类：ConfidenceParser（你当前 orchestrator 正在 import 这个类）
- 同时提供大量兼容别名方法，尽量减少后续因方法名不匹配导致的 Import/AttributeError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import ast
import json
import re


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ConfidenceParseResult:
    """
    统一解析结果对象
    """
    confidence: float = 0.0                     # 0~1
    verdict: str = ""
    issues: List[str] = field(default_factory=list)
    minimal_fix: str = ""
    raw_json: Dict[str, Any] = field(default_factory=dict)

    source_format: str = "unknown"              # json / kv / natural / mixed / unknown
    parse_ok: bool = False
    has_review_structure: bool = False

    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "verdict": self.verdict,
            "issues": self.issues,
            "minimal_fix": self.minimal_fix,
            "raw_json": self.raw_json,
            "source_format": self.source_format,
            "parse_ok": self.parse_ok,
            "has_review_structure": self.has_review_structure,
            "warnings": self.warnings,
        }


# ============================================================
# 主解析器
# ============================================================

class ConfidenceParser:
    """
    工业版置信度解析器（兼容版）

    设计目标：
    - 允许审查器输出不规整（JSON / Markdown / 键值对 / 中英混合）
    - 尽量解析出：
      confidence / verdict / issues / minimal_fix
    - 提供 should_retry / should_accept 等决策辅助
    """

    # 支持的 key 变体
    CONFIDENCE_KEYS = [
        "confidence", "conf", "score", "quality_score", "confidence_score",
        "置信度", "可信度", "评分", "分数"
    ]

    VERDICT_KEYS = [
        "verdict", "结论", "裁决", "判断"
    ]

    ISSUES_KEYS = [
        "issues", "problem", "problems", "errors", "risks", "问题", "错误", "风险点"
    ]

    MINIMAL_FIX_KEYS = [
        "minimal_fix", "fix", "repair", "suggestion", "修复建议", "建议", "最小修复", "最小改动"
    ]

    # 数字模式（支持 0.83 / 83% / "0.83/1.0"）
    RE_FLOAT = re.compile(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)")
    RE_PERCENT = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%")
    RE_FRACTION_100 = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*/\s*100(?!\d)")
    RE_FRACTION_1 = re.compile(r"(?<!\d)(\d(?:\.\d+)?)\s*/\s*1(?:\.0+)?(?!\d)")

    # Markdown / 文本标签
    RE_PHASE_TAG = re.compile(r"\[\[PHASE:[A-Z0-9_]+\]\]")

    # 常见审查字段（文本模式）
    RE_VERDICT_LINE = re.compile(r"(?:^|\n)\s*(?:verdict|结论|裁决)\s*[:：]\s*(.+?)(?=\n|$)", re.I)
    RE_ISSUES_LINE = re.compile(r"(?:^|\n)\s*(?:issues?|问题|错误|风险点)\s*[:：]\s*(.+?)(?=\n|$)", re.I)
    RE_MIN_FIX_LINE = re.compile(r"(?:^|\n)\s*(?:minimal[_\s-]*fix|修复建议|建议|最小修复|最小改动)\s*[:：]\s*(.+?)(?=\n|$)", re.I)
    RE_CONF_LINE = re.compile(r"(?:^|\n)\s*(?:confidence|置信度|评分|分数)\s*[:：]\s*(.+?)(?=\n|$)", re.I)

    # 列表项
    RE_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)、])\s*(.+?)\s*$")

    def __init__(
        self,
        default_confidence: float = 0.35,
        retry_threshold: float = 0.72,
        strong_accept_threshold: float = 0.86,
        clamp_confidence: bool = True
    ):
        self.default_confidence = float(default_confidence)
        self.retry_threshold = float(retry_threshold)
        self.strong_accept_threshold = float(strong_accept_threshold)
        self.clamp_confidence = bool(clamp_confidence)

    # ========================================================
    # 对外主入口（推荐）
    # ========================================================
    def parse_review_result(self, text: Any) -> ConfidenceParseResult:
        """
        主入口：解析审查文本，得到结构化结果
        """
        raw_text = self._normalize_text(text)
        result = ConfidenceParseResult(
            confidence=self.default_confidence,
            parse_ok=False,
            source_format="unknown"
        )

        if not raw_text.strip():
            result.warnings.append("empty_input")
            return result

        # 0) 清理阶段标签（如果审查结果意外混入 [[PHASE:...]]）
        cleaned = self._strip_phase_tags(raw_text)

        # 1) 优先尝试 JSON / 伪 JSON
        json_obj, json_mode = self._try_parse_jsonish(cleaned)
        if json_obj is not None and isinstance(json_obj, dict):
            result = self._parse_from_dict(json_obj)
            result.source_format = json_mode
            # 若 dict 解析不完整，再用文本补洞
            result = self._backfill_from_text_if_needed(result, cleaned)
            result.parse_ok = True
            return self._finalize_result(result)

        # 2) 尝试“键值对 / 文本标签”模式
        kv_result = self._parse_from_kv_text(cleaned)
        if kv_result.parse_ok:
            return self._finalize_result(kv_result)

        # 3) 尝试自然语言模式（只抓 confidence / verdict 倾向）
        natural_result = self._parse_from_natural_text(cleaned)
        return self._finalize_result(natural_result)

    # ========================================================
    # 对外兼容别名（防止 orchestrator 方法名不一致）
    # ========================================================
    def parse(self, text: Any) -> Dict[str, Any]:
        """
        常见兼容接口：返回 dict
        """
        return self.parse_review_result(text).to_dict()

    def parse_text(self, text: Any) -> Dict[str, Any]:
        return self.parse(text)

    def parse_from_text(self, text: Any) -> Dict[str, Any]:
        return self.parse(text)

    def parse_review(self, text: Any) -> Dict[str, Any]:
        return self.parse(text)

    def safe_parse(self, text: Any) -> Dict[str, Any]:
        return self.parse(text)

    def extract_confidence(self, text: Any) -> float:
        """
        仅提取置信度（兼容接口）
        """
        return self.parse_review_result(text).confidence

    def get_confidence(self, text: Any) -> float:
        return self.extract_confidence(text)

    def parse_confidence(self, text: Any) -> float:
        return self.extract_confidence(text)

    def parse_confidence_only(self, text: Any) -> float:
        return self.extract_confidence(text)

    # ========================================================
    # 决策辅助（供 orchestrator 用）
    # ========================================================
    def should_retry(
        self,
        parsed_or_text: Any,
        threshold: Optional[float] = None,
        force_retry_on_missing_structure: bool = False
    ) -> bool:
        """
        返回是否应重试
        - parsed_or_text 可以是 dict / ConfidenceParseResult / 原始文本
        """
        result = self._coerce_to_result(parsed_or_text)
        th = self.retry_threshold if threshold is None else float(threshold)

        # 若要求结构化审查，但没解析到结构，允许强制重试
        if force_retry_on_missing_structure and not result.has_review_structure:
            return True

        # 置信度不足 -> 重试
        if result.confidence < th:
            return True

        # verdict 明确否定时 -> 重试
        verdict_low = result.verdict.strip().lower()
        negative_tokens = ["reject", "fail", "incorrect", "wrong", "unsafe", "不通过", "错误", "不正确", "有误"]
        if any(tok in verdict_low for tok in negative_tokens):
            return True

        return False

    def should_accept(
        self,
        parsed_or_text: Any,
        threshold: Optional[float] = None
    ) -> bool:
        result = self._coerce_to_result(parsed_or_text)
        th = self.strong_accept_threshold if threshold is None else float(threshold)
        return result.confidence >= th

    def build_retry_decision(
        self,
        parsed_or_text: Any,
        threshold: Optional[float] = None,
        force_retry_on_missing_structure: bool = False
    ) -> Dict[str, Any]:
        """
        返回完整决策信息（兼容有些 orchestrator 喜欢拿 dict）
        """
        result = self._coerce_to_result(parsed_or_text)
        retry = self.should_retry(
            result,
            threshold=threshold,
            force_retry_on_missing_structure=force_retry_on_missing_structure
        )
        return {
            "retry": retry,
            "confidence": result.confidence,
            "verdict": result.verdict,
            "has_review_structure": result.has_review_structure,
            "issues_count": len(result.issues),
            "minimal_fix": result.minimal_fix,
            "parsed": result.to_dict()
        }

    def summarize_for_alpha(self, parsed_or_text: Any) -> str:
        """
        把审查结果清洗成 Alpha 易消费的文本（兼容你前面多模型终审链路）
        """
        result = self._coerce_to_result(parsed_or_text)

        lines: List[str] = []
        lines.append(f"Confidence: {result.confidence:.3f}")

        if result.verdict:
            lines.append(f"Verdict: {result.verdict}")

        if result.issues:
            lines.append("Issues:")
            for item in result.issues:
                lines.append(f"- {item}")

        if result.minimal_fix:
            lines.append(f"Minimal Fix: {result.minimal_fix}")

        if not result.verdict and not result.issues and not result.minimal_fix:
            lines.append("Issues: 审查器未给出结构化字段，建议谨慎重审。")

        return "\n".join(lines)

    # ========================================================
    # 解析分支：JSON / 伪 JSON
    # ========================================================
    def _try_parse_jsonish(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        尝试解析：
        1) 原生 JSON
        2) Markdown ```json 代码块中的 JSON
        3) Python dict 样式（ast.literal_eval）
        4) 从文本中提取大括号块
        """
        text = text.strip()

        # 1) 直接 JSON
        obj = self._json_loads_safe(text)
        if isinstance(obj, dict):
            return obj, "json"

        # 2) 提取 markdown json code block
        for block in self._extract_code_blocks(text):
            obj = self._json_loads_safe(block)
            if isinstance(obj, dict):
                return obj, "json_codeblock"

        # 3) 提取大括号块再解析
        for cand in self._extract_brace_objects(text):
            obj = self._json_loads_safe(cand)
            if isinstance(obj, dict):
                return obj, "json_embedded"
            obj2 = self._literal_eval_dict_safe(cand)
            if isinstance(obj2, dict):
                return obj2, "python_dict_embedded"

        # 4) 文本整体当 Python dict
        obj = self._literal_eval_dict_safe(text)
        if isinstance(obj, dict):
            return obj, "python_dict"

        return None, "unknown"

    def _parse_from_dict(self, obj: Dict[str, Any]) -> ConfidenceParseResult:
        result = ConfidenceParseResult()
        result.raw_json = obj
        result.parse_ok = True
        result.has_review_structure = False

        # confidence
        conf_val = self._pick_first_key(obj, self.CONFIDENCE_KEYS)
        if conf_val is not None:
            parsed_conf = self._parse_confidence_value(conf_val)
            if parsed_conf is not None:
                result.confidence = parsed_conf

        # verdict
        verdict_val = self._pick_first_key(obj, self.VERDICT_KEYS)
        if verdict_val is not None:
            result.verdict = self._stringify_simple(verdict_val).strip()

        # issues
        issues_val = self._pick_first_key(obj, self.ISSUES_KEYS)
        result.issues = self._normalize_issues(issues_val)

        # minimal fix
        fix_val = self._pick_first_key(obj, self.MINIMAL_FIX_KEYS)
        if fix_val is not None:
            result.minimal_fix = self._stringify_simple(fix_val).strip()

        result.has_review_structure = bool(result.verdict or result.issues or result.minimal_fix)
        return result

    # ========================================================
    # 解析分支：键值文本
    # ========================================================
    def _parse_from_kv_text(self, text: str) -> ConfidenceParseResult:
        result = ConfidenceParseResult(
            confidence=self.default_confidence,
            source_format="kv",
            parse_ok=False
        )

        conf_line = self._match_group(self.RE_CONF_LINE, text)
        verdict_line = self._match_group(self.RE_VERDICT_LINE, text)
        issues_line = self._match_group(self.RE_ISSUES_LINE, text)
        min_fix_line = self._match_group(self.RE_MIN_FIX_LINE, text)

        if conf_line:
            conf = self._parse_confidence_from_free_text(conf_line)
            if conf is not None:
                result.confidence = conf

        if verdict_line:
            result.verdict = verdict_line.strip()

        if issues_line:
            # Issues 行里可能是 "a; b; c" 或 "['a','b']"
            result.issues = self._parse_issues_from_line(issues_line)

        if min_fix_line:
            result.minimal_fix = min_fix_line.strip()

        # 如果 Issues 不是单行，而是下面跟着 bullet list，尝试补提取
        if not result.issues:
            result.issues = self._extract_issue_bullets(text)

        result.has_review_structure = bool(result.verdict or result.issues or result.minimal_fix)
        result.parse_ok = result.has_review_structure or (conf_line is not None)
        return result

    # ========================================================
    # 解析分支：自然语言
    # ========================================================
    def _parse_from_natural_text(self, text: str) -> ConfidenceParseResult:
        result = ConfidenceParseResult(
            confidence=self.default_confidence,
            source_format="natural",
            parse_ok=True
        )

        # 1) 置信度
        conf = self._parse_confidence_from_free_text(text)
        if conf is not None:
            result.confidence = conf

        # 2) verdict 倾向（弱规则）
        low = text.lower()
        if any(x in low for x in ["looks good", "acceptable", "pass", "通过", "可接受", "基本正确"]):
            result.verdict = "通过/可接受（自然语言推断）"
        elif any(x in low for x in ["incorrect", "wrong", "unsafe", "不通过", "有误", "错误", "风险较高"]):
            result.verdict = "存在问题（自然语言推断）"

        # 3) 尝试抓一行建议
        m_fix = self._match_group(self.RE_MIN_FIX_LINE, text)
        if m_fix:
            result.minimal_fix = m_fix.strip()

        # 4) issues（若出现 bullet）
        issues = self._extract_issue_bullets(text)
        if issues:
            result.issues = issues

        result.has_review_structure = bool(result.verdict or result.issues or result.minimal_fix)
        return result

    # ========================================================
    # 回填（dict 解析成功但字段缺失）
    # ========================================================
    def _backfill_from_text_if_needed(self, result: ConfidenceParseResult, text: str) -> ConfidenceParseResult:
        if result.verdict and result.issues and result.minimal_fix:
            return result

        # 补 confidence（防止 dict 里没有标准字段）
        if result.confidence <= 0:
            conf = self._parse_confidence_from_free_text(text)
            if conf is not None:
                result.confidence = conf

        # 补 verdict / issues / fix
        if not result.verdict:
            v = self._match_group(self.RE_VERDICT_LINE, text)
            if v:
                result.verdict = v.strip()

        if not result.issues:
            issues_line = self._match_group(self.RE_ISSUES_LINE, text)
            if issues_line:
                result.issues = self._parse_issues_from_line(issues_line)
            if not result.issues:
                result.issues = self._extract_issue_bullets(text)

        if not result.minimal_fix:
            f = self._match_group(self.RE_MIN_FIX_LINE, text)
            if f:
                result.minimal_fix = f.strip()

        result.has_review_structure = bool(result.verdict or result.issues or result.minimal_fix)
        return result

    # ========================================================
    # 最终收尾
    # ========================================================
    def _finalize_result(self, result: ConfidenceParseResult) -> ConfidenceParseResult:
        # 置信度归一化
        result.confidence = self._normalize_confidence(result.confidence)

        # 字段清洗
        result.verdict = self._clean_inline_text(result.verdict)
        result.minimal_fix = self._clean_inline_text(result.minimal_fix)

        cleaned_issues = []
        for item in result.issues:
            s = self._clean_inline_text(item)
            if s:
                cleaned_issues.append(s)
        result.issues = cleaned_issues

        # parse_ok 最低保证
        if not result.parse_ok:
            result.parse_ok = True

        # 如果完全无结构且无 confidence，给 warning
        if not result.has_review_structure and result.confidence == self.default_confidence:
            result.warnings.append("fallback_default_confidence")

        return result

    # ========================================================
    # 类型适配
    # ========================================================
    def _coerce_to_result(self, parsed_or_text: Any) -> ConfidenceParseResult:
        if isinstance(parsed_or_text, ConfidenceParseResult):
            return parsed_or_text

        if isinstance(parsed_or_text, dict):
            # 尝试从 dict 恢复
            result = ConfidenceParseResult()
            result.confidence = self._normalize_confidence(
                self._safe_float(parsed_or_text.get("confidence", self.default_confidence), self.default_confidence)
            )
            result.verdict = self._clean_inline_text(str(parsed_or_text.get("verdict", "") or ""))
            result.minimal_fix = self._clean_inline_text(str(parsed_or_text.get("minimal_fix", "") or ""))

            issues = parsed_or_text.get("issues", [])
            if isinstance(issues, list):
                result.issues = [self._clean_inline_text(str(x)) for x in issues if str(x).strip()]
            elif isinstance(issues, str) and issues.strip():
                result.issues = self._parse_issues_from_line(issues)

            result.raw_json = parsed_or_text.get("raw_json", {}) if isinstance(parsed_or_text.get("raw_json"), dict) else {}
            result.source_format = str(parsed_or_text.get("source_format", "dict"))
            result.parse_ok = bool(parsed_or_text.get("parse_ok", True))
            result.has_review_structure = bool(
                parsed_or_text.get("has_review_structure", bool(result.verdict or result.issues or result.minimal_fix))
            )
            return result

        # 否则当原始文本解析
        return self.parse_review_result(parsed_or_text)

    # ========================================================
    # 置信度解析核心
    # ========================================================
    def _parse_confidence_value(self, value: Any) -> Optional[float]:
        """
        输入可能是数值/字符串/百分比
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            v = float(value)
            # 0~1 正常；>1 且 <=100 当百分比
            if 0 <= v <= 1:
                return self._normalize_confidence(v)
            if 1 < v <= 100:
                return self._normalize_confidence(v / 100.0)
            return self._normalize_confidence(v)

        # 字符串
        return self._parse_confidence_from_free_text(str(value))

    def _parse_confidence_from_free_text(self, text: str) -> Optional[float]:
        if not text:
            return None

        s = text.strip()

        # 1) 百分比优先（83%）
        m = self.RE_PERCENT.search(s)
        if m:
            v = self._safe_float(m.group(1), None)
            if v is not None:
                return self._normalize_confidence(v / 100.0)

        # 2) x/100
        m = self.RE_FRACTION_100.search(s)
        if m:
            v = self._safe_float(m.group(1), None)
            if v is not None:
                return self._normalize_confidence(v / 100.0)

        # 3) x/1.0
        m = self.RE_FRACTION_1.search(s)
        if m:
            v = self._safe_float(m.group(1), None)
            if v is not None:
                return self._normalize_confidence(v)

        # 4) 纯 0~1 浮点
        m = self.RE_FLOAT.search(s)
        if m:
            v = self._safe_float(m.group(1), None)
            if v is not None:
                return self._normalize_confidence(v)

        # 5) 有些会写“置信度高/中/低”
        low = s.lower()
        if any(tok in low for tok in ["高", "high", "较高"]):
            return 0.85
        if any(tok in low for tok in ["中", "medium", "moderate", "一般"]):
            return 0.65
        if any(tok in low for tok in ["低", "low", "较低"]):
            return 0.45

        return None

    def _normalize_confidence(self, v: Any) -> float:
        f = self._safe_float(v, self.default_confidence)
        if self.clamp_confidence:
            if f < 0:
                f = 0.0
            if f > 1:
                # 若被传入 75 这种值，按百分比兜底
                if f <= 100:
                    f = f / 100.0
                else:
                    f = 1.0
        return round(float(f), 4)

    # ========================================================
    # issues 解析
    # ========================================================
    def _normalize_issues(self, value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, list):
            out = []
            for x in value:
                s = self._stringify_simple(x).strip()
                if s:
                    out.append(s)
            return out

        if isinstance(value, tuple):
            return [self._stringify_simple(x).strip() for x in value if self._stringify_simple(x).strip()]

        if isinstance(value, str):
            return self._parse_issues_from_line(value)

        # dict / 其他对象 -> 一条字符串
        s = self._stringify_simple(value).strip()
        return [s] if s else []

    def _parse_issues_from_line(self, text: str) -> List[str]:
        if not text:
            return []

        s = text.strip()

        # 尝试解析成 list 字面量
        obj = self._literal_eval_safe(s)
        if isinstance(obj, list):
            out = []
            for x in obj:
                xs = self._stringify_simple(x).strip()
                if xs:
                    out.append(xs)
            if out:
                return out

        # 分隔符拆分
        # 支持：; /； / | / 、 / 换行
        parts = re.split(r"[;\n；|]+", s)
        cleaned = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # 再拆一次 “1) xxx 2) yyy” 这种紧凑格式不太可靠，这里不硬拆
            cleaned.append(p)

        if cleaned:
            return cleaned

        return [s] if s else []

    def _extract_issue_bullets(self, text: str) -> List[str]:
        if not text:
            return []

        lines = text.splitlines()
        out: List[str] = []

        in_issue_zone = False
        for line in lines:
            low = line.strip().lower()

            # 进入 issue 区域
            if re.match(r"^(issues?|问题|错误|风险点)\s*[:：]?\s*$", low, re.I):
                in_issue_zone = True
                continue

            # 遇到另一个大字段时退出
            if in_issue_zone and re.match(r"^(verdict|结论|裁决|minimal[_\s-]*fix|修复建议|建议)\s*[:：]", low, re.I):
                in_issue_zone = False

            if in_issue_zone:
                m = self.RE_BULLET.match(line)
                if m:
                    item = self._clean_inline_text(m.group(1))
                    if item:
                        out.append(item)

        return out

    # ========================================================
    # 通用提取 / 解析工具
    # ========================================================
    def _pick_first_key(self, obj: Dict[str, Any], keys: List[str]) -> Any:
        if not isinstance(obj, dict):
            return None

        # 先精确 key
        for k in keys:
            if k in obj:
                return obj[k]

        # 再大小写不敏感 / 下划线连字符归一
        normalized_map = {}
        for key in obj.keys():
            if not isinstance(key, str):
                continue
            nk = self._norm_key(key)
            normalized_map[nk] = key

        for k in keys:
            nk = self._norm_key(k)
            if nk in normalized_map:
                return obj[normalized_map[nk]]

        return None

    def _norm_key(self, key: str) -> str:
        return re.sub(r"[\s_\-]+", "", str(key).strip().lower())

    def _json_loads_safe(self, text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except Exception:
            return None

    def _literal_eval_safe(self, text: str) -> Optional[Any]:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    def _literal_eval_dict_safe(self, text: str) -> Optional[Dict[str, Any]]:
        obj = self._literal_eval_safe(text)
        return obj if isinstance(obj, dict) else None

    def _extract_code_blocks(self, text: str) -> List[str]:
        # 提取 ```json ... ``` / ``` ... ```
        blocks = []
        for m in re.finditer(r"```(?:json|python|txt)?\s*([\s\S]*?)```", text, re.I):
            block = (m.group(1) or "").strip()
            if block:
                blocks.append(block)
        return blocks

    def _extract_brace_objects(self, text: str) -> List[str]:
        """
        粗略提取 {...} 块（不做完整语法栈解析，但足够应对大多数场景）
        """
        candidates = []
        stack = []
        start = None

        for i, ch in enumerate(text):
            if ch == "{":
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidates.append(text[start:i+1])
                        start = None
        return candidates

    def _match_group(self, pattern: re.Pattern, text: str) -> Optional[str]:
        m = pattern.search(text or "")
        if not m:
            return None
        try:
            return m.group(1)
        except Exception:
            return None

    def _strip_phase_tags(self, text: str) -> str:
        return self.RE_PHASE_TAG.sub("", text or "")

    def _normalize_text(self, text: Any) -> str:
        if text is None:
            return ""
        s = str(text)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return s

    def _stringify_simple(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False)
        except Exception:
            pass
        return str(value)

    def _clean_inline_text(self, s: str) -> str:
        if not s:
            return ""
        s = str(s).strip()

        # 去掉 markdown 强调符号外壳
        s = re.sub(r"^\*\*(.+?)\*\*$", r"\1", s)

        # 去掉多余引号包裹
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()

        # 压缩空白
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _safe_float(self, x: Any, default: Optional[float]) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return default


# ============================================================
# 兼容性模块级函数（防止旧代码不是 class 调用）
# ============================================================

_default_parser = ConfidenceParser()

def parse_confidence(text: Any) -> float:
    return _default_parser.parse_confidence(text)

def parse_review(text: Any) -> Dict[str, Any]:
    return _default_parser.parse_review(text)

def should_retry(parsed_or_text: Any, threshold: Optional[float] = None) -> bool:
    return _default_parser.should_retry(parsed_or_text, threshold=threshold)

def summarize_for_alpha(parsed_or_text: Any) -> str:
    return _default_parser.summarize_for_alpha(parsed_or_text)


__all__ = [
    "ConfidenceParseResult",
    "ConfidenceParser",
    "parse_confidence",
    "parse_review",
    "should_retry",
    "summarize_for_alpha",
]