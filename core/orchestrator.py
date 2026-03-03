# core/orchestrator.py
import asyncio
import json as _json
import re as _re
import time as _time
import traceback


class StageTimeoutError(Exception):
    """某阶段在限定时间内未完成（模型卡住或过慢）"""
    def __init__(self, stage_name: str, limit_seconds: float):
        self.stage_name = stage_name
        self.limit_seconds = limit_seconds


from core.config import (
    MODEL_ALPHA, MODEL_BETA, MODEL_GAMMA, MODEL_VISION,
    CONFIDENCE_THRESHOLD, MAX_RETRY_COUNT,
    VISION_STAGE_TIMEOUT_SECONDS, VISION_STAGE_MAX_RETRY,
    STAGE_RETRY_BACKOFF_SECONDS, ENABLE_L0_QUALITY_GUARD,
    ENABLE_BETA_REVIEW, ENABLE_GAMMA_JUDGE,
    ALPHA_STAGE_TIMEOUT_SECONDS, BETA_STAGE_TIMEOUT_SECONDS, GAMMA_STAGE_TIMEOUT_SECONDS,
)
from core.alpha_engine import AlphaEngine
from core.beta_engine import BetaEngine
from core.gamma_engine import GammaEngine
from core.oculus_engine import OculusEngine
from core.stage_manager import StageManager
from core.retry_controller import RetryController
from core.confidence_parser import ConfidenceParser
from core.error_types import ModelExecutionError, JsonParseError
from core.memory import MemoryManager
from core.context_trimmer import ContextTrimmer
from core.router import RiskRouter
from core.protocol import OllamaStreamExecutor

_INTERNAL_TAG_RE = _re.compile(
    r"\[TaskPack\].*?$|^RISK\s*=\s*L\d.*?$|\[/?VisionPack\]|"
    r"^\s*\[(上下文背景?|用户(当前)?问题|回答规则|你的角色)\].*?$|"
    r"^\s*(Confidence|Verdict|Issues|Minimal_Fix)\s*[:：].*?$|"
    r"^\s*(Problem|Given|Goal|Constraints|Ambiguities|What_to_verify):.*?$",
    _re.MULTILINE | _re.IGNORECASE
)
# thinking 标签单独处理，支持不闭合/嵌套/格式不标准
_THINKING_RE = _re.compile(
    r"<thinking>[\s\S]*?</thinking>|"       # 标准闭合
    r"<thinking>[\s\S]*$|"                  # 未闭合（到文末）
    r"<think>[\s\S]*?</think>|"             # 变体标签
    r"<think>[\s\S]*$",                     # 变体未闭合
    _re.IGNORECASE
)

# ==========================================
# System Prompts（原生 Ollama system 字段）
# ==========================================
ALPHA_SYSTEM = """你是 Trinitas-Alpha，一个专业的 AI 助手，拥有独立判断能力。
你必须用中文回答。你不是一个只会顺从的工具——如果用户的问题本身有问题，你必须指出。

回答规范：
- 结论先行，开篇给核心结论
- 用 ## 和 ### 分模块
- 用 **加粗** 强调关键信息
- 多要点用列表
- 代码用 ```语言名 包裹
- 专业名词旁附通俗解释
- 禁止暴露内部标签（TaskPack/RISK/Alpha/Beta/Gamma等）"""

BETA_SYSTEM = """你是 Trinitas-Beta，一个独立的逻辑审查员。你必须先对问题做独立分析并给出自己的解决方案，再对比 Alpha 的回答找出错误和盲区。
你必须用中文输出。你必须严格按 JSON 格式输出审查结果。"""

GAMMA_SYSTEM = """你是 Trinitas-Gamma，最终裁判。你需要在参考 Beta 审查的基础上，独立再做一次对回答质量与正确性的判断，综合给出裁决。
你必须用中文输出。你必须严格按 JSON 格式输出裁决结果。"""

# ==========================================
# Few-shot 示例
# ==========================================
BETA_FEWSHOT = """
以下是一个审查输出的示例（先独立审题与独立方案，再对比 Alpha）：

用户问题："用纸做一把能切牛排的刀"
Alpha回答："可以将A4纸反复折叠增加硬度..."

你的审查输出：
```json
{
  "step1_审题": "用户想用纸做切割工具，但纸的硬度和耐久性远不够切牛排",
  "step2_独立方案": "应先判断纸刀是否可行；不可行则明确否定前提，再建议陶瓷刀或不锈钢刀等可行方案",
  "step3_对比": "Alpha没有指出纸刀根本无法胜任切割任务，只是在讨论如何折叠，被问题带偏了",
  "step4_建议": "应先指出纸刀不可行，再建议用陶瓷刀或不锈钢刀",
  "verdict": "FAIL",
  "issues": ["纸的抗拉强度不足以切割肉类纤维", "未指出方案的根本不可行性"],
  "improvements": ["先否定不可行前提再给替代方案", "明确物理约束"],
  "fix": "应先否定前提，再给替代方案"
}
```

现在请审查以下内容：
"""

GAMMA_FEWSHOT = """
以下是一个裁决输出的示例（含独立审视）：

```json
{
  "step1_理解": "用户想了解如何在家搭建NAS存储",
  "step2_可行性": "回答推荐的方案硬件成本合理，步骤清晰可执行",
  "step3_风险": "未提及数据备份策略和UPS断电保护，有数据丢失风险",
  "step4_独立审视": "从终审角度，回答可用但缺关键生产环境要素，应补充备份与断电保护才能判为完整",
  "verdict": "FAIL",
  "confidence": 0.55,
  "issues": ["缺少数据备份方案", "未提及UPS/断电保护"],
  "improvements": ["补充3-2-1备份策略", "增加UPS/断电保护建议"],
  "fix": "补充3-2-1备份策略和UPS建议"
}
```

现在请裁决以下内容：
"""


class Orchestrator:

    def __init__(self):
        executor = OllamaStreamExecutor()
        self.alpha_engine = AlphaEngine(executor, MODEL_ALPHA)
        self.beta_engine = BetaEngine(executor, MODEL_BETA)
        self.gamma_engine = GammaEngine(executor, MODEL_GAMMA)
        self.oculus_engine = OculusEngine(executor, MODEL_VISION)
        self.stage_manager = StageManager()
        self.retry_controller = RetryController(max_retry=MAX_RETRY_COUNT)
        self.confidence_parser = ConfidenceParser()
        self.memory = MemoryManager()
        self.trimmer = ContextTrimmer()
        self.router = RiskRouter()
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self._token_stats_re = _re.compile(r"\n?\[\[TOKEN_STATS:(.*?)\]\]\n?")

    def _strip_token_stats(self, text):
        stats = []
        for m in self._token_stats_re.finditer(text):
            try: stats.append(_json.loads(m.group(1)))
            except: pass
        return self._token_stats_re.sub("", text), stats

    @staticmethod
    def _merge_token_stats(all_stats):
        merged = {"total_eval_count": 0, "total_prompt_eval_count": 0, "total_eval_duration_ns": 0}
        for s in all_stats:
            merged["total_eval_count"] += s.get("eval_count", 0)
            merged["total_prompt_eval_count"] += s.get("prompt_eval_count", 0)
            merged["total_eval_duration_ns"] += s.get("eval_duration_ns", 0)
        if merged["total_eval_duration_ns"] > 0 and merged["total_eval_count"] > 0:
            merged["avg_tokens_per_second"] = round(merged["total_eval_count"] / (merged["total_eval_duration_ns"] / 1e9), 2)
        return merged

    def _phase(self, name): return f"\n[[PHASE:{name}]]\n"

    @staticmethod
    def _clean_final_answer(text):
        cleaned = _THINKING_RE.sub("", text)
        cleaned = _INTERNAL_TAG_RE.sub("", cleaned)
        return _re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    async def _build_context(self, chat_id):
        self.memory.ensure_chat(chat_id)
        await self.memory.maybe_compress(chat_id)
        lt = self.memory.get_long_term_summary(chat_id)
        rows = self.memory.get_recent_messages(chat_id)
        rt = "\n".join([f"{r}:{c}" for r, c in rows])
        return self.trimmer.trim(lt, rt)

    async def _run_vision_and_merge(self, user_message, image_bytes):
        vr = await self.oculus_engine.extract_from_bytes(image_bytes=image_bytes)
        if user_message and user_message.strip():
            return f"用户问题：{user_message.strip()}\n\n视觉识别结果：\n{vr}", vr
        return f"请基于以下视觉识别结果进行回答：\n{vr}", vr

    # ==========================================
    # Prompts（用户内容 = prompt，角色指令 = system）
    # ==========================================
    def _build_alpha_prompt(self, context_text, working_message, risk_level):
        cot = """请先在 <thinking> 标签中完成你的推理过程，然后在 </thinking> 之后给出最终答案。
<thinking> 中的内容不会展示给用户，你可以自由思考。"""
        intent_step = """
【步骤 0：判断用户意图】
先判断本条消息是否「仅包含打招呼、寒暄、随口一问」而没有任何需要解答的具体问题或任务。
例如：你好、在吗、你在干嘛、干嘛呢、忙不忙、吃了吗、最近怎么样、在不在、嗨、hello、hi 等，或仅有类似含义的短句。
- 若是：只需用一两句自然、友好地回应，并邀请用户说出具体需求。不要输出问题诊断、审题清单、替代方案或需求挖掘话术。
- 若否（用户已提出具体问题、任务或请求）：继续下面的步骤 1。"""
        trap_check = """
【步骤 1：先审题再答题】用以下清单审视问题：
A)事实前提错误 B)违反自然规律 C)概念混淆 D)隐含矛盾 E)条件缺失
F)幸存者偏差 G)规模谬误 H)忽略副作用 I)成本不现实 J)伪问题
命中则先指出问题，再给替代方案。"""
        rules = "致命缺陷必须指出。多方案按可行性排列。不确定处标注待验证。"
        if risk_level >= 2:
            rules += " 高复杂度多角度验证。涉及代码给可执行版本。"
        return f"{cot}\n\n{intent_step}\n\n{trap_check}\n\n{rules}\n\n[上下文]\n{context_text}\n\n[用户问题]\n{working_message}\n\n请按步骤 0 判断后，再决定是简短回应还是执行步骤 1 审题并回答。"

    def _build_beta_prompt(self, user_question, alpha_answer):
        return f"""{BETA_FEWSHOT}
[用户原始问题]
{user_question}

[Alpha的回答]
{alpha_answer[:2500]}

请严格输出 JSON 格式（不要输出其他内容）。先完成 step1 审题与 step2 独立方案，再 step3 对比 Alpha、step4 建议：
```json
{{
  "step1_审题": "（你对问题本身的独立分析，1-2句）",
  "step2_独立方案": "（你针对该问题自己给出的独立解决思路或关键要点，1-3句）",
  "step3_对比": "（Alpha的回答有什么问题或遗漏？1-3句）",
  "step4_建议": "（可执行修正方向，或写 Alpha分析全面）",
  "verdict": "PASS 或 FAIL",
  "issues": ["问题1", "问题2"],
  "improvements": ["改进建议1", "改进建议2"],
  "fix": "（修正方向）"
}}
```"""

    def _build_gamma_prompt(self, user_question, alpha_answer, beta_review):
        beta_section = f"\n[Beta审查结果]\n{beta_review[:1200]}\n" if beta_review.strip() else ""
        return f"""{GAMMA_FEWSHOT}
[用户原始问题]
{user_question}

[Alpha的回答]
{alpha_answer[:2500]}
{beta_section}
请参考 Beta 审查结果，并在此基础上做 step4 独立审视后再裁决。严格输出 JSON 格式：
```json
{{
  "step1_理解": "（用户核心需求，1句）",
  "step2_可行性": "（按这个回答执行能成功吗？1-2句）",
  "step3_风险": "（最坏情况，1-2句）",
  "step4_独立审视": "（你作为终审的独立判断：回答是否完整、正确、可执行，1-3句）",
  "verdict": "PASS 或 FAIL",
  "confidence": 0.0到1.0,
  "issues": ["问题1"],
  "improvements": ["改进建议1"],
  "fix": "（修正方向或写无）"
}}
```"""

    def _build_alpha_rewrite_prompt(self, context_text, user_question, original, feedback):
        return f"""[上下文]
{context_text}

[用户问题]
{user_question}

[你之前的回答]
{original[:1500]}

[专家审查反馈]
{feedback}

请根据反馈重新生成修正后的完整答案。直接给出答案，不解释修改了什么。"""

    # ==========================================
    # Heartbeat
    # ==========================================
    async def _with_heartbeat(self, async_gen, interval=15.0, hb="\u200b"):
        async def nxt(): return await anext(async_gen)
        task = None
        while True:
            if task is None: task = asyncio.create_task(nxt())
            done, _ = await asyncio.wait([task], timeout=interval)
            if task in done:
                try: chunk = task.result(); yield False, chunk; task = None
                except StopAsyncIteration: break
                except Exception as e: raise e
            else: yield True, hb

    async def _collect_with_heartbeat(self, engine, prompt, all_stats, system=None):
        """收集流式输出为完整文本（内部缓冲），同时 yield 心跳防超时"""
        buf = ""
        async for is_hb, chunk in self._with_heartbeat(engine.execute(prompt=prompt, system=system)):
            if is_hb:
                yield chunk  # 心跳透传给前端，防止连接断开
            else:
                c, s = self._strip_token_stats(chunk); all_stats.extend(s); buf += c
        self._last_collected = buf  # 通过实例变量传回结果

    async def _stream_with_stage_timeout(self, aiter, stage_name: str, stage_seconds: float):
        """包装异步迭代器：单阶段总时长超过 stage_seconds 则终止并抛出 StageTimeoutError"""
        queue = asyncio.Queue()
        async def put_all():
            try:
                async for c in aiter:
                    await queue.put(c)
            finally:
                await queue.put(None)
        task = asyncio.create_task(put_all())
        start = _time.monotonic()
        idle_wait = 120  # 单次等待 chunk 的最长时间（秒）
        try:
            while True:
                remaining = stage_seconds - (_time.monotonic() - start)
                if remaining <= 0:
                    task.cancel()
                    try: await task
                    except asyncio.CancelledError: pass
                    yield f"\n[阶段超时] {stage_name} 在 {stage_seconds} 秒内未完成，已终止。请简化问题或稍后重试。\n"
                    raise StageTimeoutError(stage_name, stage_seconds)
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=min(idle_wait, remaining))
                except asyncio.TimeoutError:
                    continue
                if chunk is None:
                    break
                yield chunk
        finally:
            try: task.cancel()
            except Exception: pass

    def _build_stats(self, t0, timings, stats):
        elapsed = round(_time.monotonic() - t0, 2)
        m = self._merge_token_stats(stats)
        return {"total_time_s": elapsed, "phase_times": timings,
                "total_eval_tokens": m.get("total_eval_count", 0),
                "total_prompt_tokens": m.get("total_prompt_eval_count", 0),
                "total_tokens": m.get("total_eval_count", 0) + m.get("total_prompt_eval_count", 0),
                "avg_speed_tps": m.get("avg_tokens_per_second", 0)}

    def _parse_json_review(self, raw):
        """从审查文本中提取 JSON，多重容错"""
        if not raw or not raw.strip():
            return None

        # 尝试1: ```json ... ``` 代码块
        m = _re.search(r"```json\s*([\s\S]*?)```", raw)
        if m:
            try: return _json.loads(m.group(1))
            except: pass

        # 尝试2: ``` ... ``` 无语言标注
        m = _re.search(r"```\s*([\s\S]*?)```", raw)
        if m:
            try: return _json.loads(m.group(1))
            except: pass

        # 尝试3: 裸 { ... } 块
        m = _re.search(r"\{[\s\S]*\}", raw)
        if m:
            try: return _json.loads(m.group(0))
            except: pass

        # 尝试4: 修复常见 JSON 错误（中文引号、尾逗号）再解析
        if "{" in raw:
            fixed = raw[raw.index("{"):raw.rindex("}") + 1] if "}" in raw else ""
            if fixed:
                fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')  # 中文引号
                fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
                fixed = _re.sub(r",\s*([}\]])", r"\1", fixed)  # 尾逗号
                try: return _json.loads(fixed)
                except: pass

        # 尝试5: 从文本中提取 verdict（回退到非 JSON 解析）
        verdict = "PASS"
        if "FAIL" in raw.upper():
            verdict = "FAIL"
        issues = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- ") and len(stripped) > 4:
                issues.append(stripped[2:])
        if verdict == "FAIL" or issues:
            return {"verdict": verdict, "issues": issues, "fix": "", "confidence": 0.4 if verdict == "FAIL" else 0.7}
        return None

    # ==========================================
    # Main
    # ==========================================
    async def handle(self, chat_id, message, image_bytes=None, mode="auto"):
        try:
            user_message = message or ""
            t0 = _time.monotonic()
            all_stats, timings = [], {}

            self.memory.ensure_chat(chat_id)
            self.memory.save_message(chat_id, "用户", f"[图片] {user_message}".strip() if image_bytes else user_message)

            # ===== Vision =====
            if image_bytes is not None:
                yield self._phase("OCULUS_VISION_ANALYSIS")
                try:
                    merged_msg = None
                    for attempt in range(VISION_STAGE_MAX_RETRY + 1):
                        try:
                            task = asyncio.create_task(self._run_vision_and_merge(user_message, image_bytes))
                            started = asyncio.get_running_loop().time()
                            while not task.done():
                                if asyncio.get_running_loop().time() - started > float(VISION_STAGE_TIMEOUT_SECONDS):
                                    task.cancel(); raise TimeoutError("视觉超时")
                                done, _ = await asyncio.wait([task], timeout=15.0)
                                if not done: yield "\u200b"
                            merged_msg, raw_vision = task.result(); break
                        except Exception as e:
                            if attempt < VISION_STAGE_MAX_RETRY:
                                await asyncio.sleep(STAGE_RETRY_BACKOFF_SECONDS * (attempt + 1)); continue
                            raise e
                    if merged_msg is None: raise RuntimeError("视觉无结果")
                    yield raw_vision + "\n"; working_message = merged_msg
                except Exception as e:
                    self.memory.save_message(chat_id, "AI", f"[视觉异常]"); yield f"[视觉异常]: {e}"; return
            else:
                working_message = user_message

            # ===== Route =====
            risk = self.router.analyze(working_message, has_image=bool(image_bytes))

            # 功能开关生效
            if not ENABLE_BETA_REVIEW: risk.need_beta = False
            if not ENABLE_GAMMA_JUDGE: risk.need_gamma = False

            if mode == "pro":
                risk.need_beta = True; risk.need_gamma = True
                if risk.level < 2: risk.level = 2
            elif mode == "instant":
                risk.need_beta = False; risk.need_gamma = False

            label = {"auto": "Auto", "pro": "Pro", "instant": "Instant"}.get(mode, mode)
            yield self._phase("ROUTER_DECISION")
            yield f"L{risk.level} | Beta={risk.need_beta} | Gamma={risk.need_gamma} | Mode={label}\n"

            context_text = await self._build_context(chat_id)
            need_review = risk.need_beta or risk.need_gamma

            # ===== Alpha 生成 =====
            yield self._phase("ALPHA_GENERATION")
            alpha_prompt = self._build_alpha_prompt(context_text, working_message, risk.level)
            tp = _time.monotonic()

            if need_review:
                # 审查模式：Alpha 内部缓冲，但心跳透传给前端防超时；带阶段超时
                self._last_collected = ""
                async for hb_chunk in self._stream_with_stage_timeout(
                    self._collect_with_heartbeat(self.alpha_engine, alpha_prompt, all_stats, system=ALPHA_SYSTEM),
                    "Alpha", ALPHA_STAGE_TIMEOUT_SECONDS
                ):
                    yield hb_chunk
                alpha_draft = self._last_collected
                yield f"Alpha 草稿已生成（{len(alpha_draft)} 字），进入审查...\n"
            else:
                # Instant 模式：直接流式输出，带阶段超时
                alpha_draft = ""
                async for item in self._stream_with_stage_timeout(
                    self._with_heartbeat(self.alpha_engine.execute(prompt=alpha_prompt, system=ALPHA_SYSTEM)),
                    "Alpha", ALPHA_STAGE_TIMEOUT_SECONDS
                ):
                    is_hb, chunk = item
                    if is_hb: yield chunk
                    else:
                        c, s = self._strip_token_stats(chunk); all_stats.extend(s); alpha_draft += c; yield c
            timings["ALPHA"] = round(_time.monotonic() - tp, 2)

            # ===== No review → output directly =====
            if not need_review:
                final = self._clean_final_answer(alpha_draft)
                self.memory.save_message(chat_id, "AI", final)
                yield self._phase("FINAL"); yield final
                yield f"\n[[TRINITAS_STATS:{_json.dumps(self._build_stats(t0, timings, all_stats))}]]\n"
                return

            # ===== Beta 审查（先跑 Beta，Gamma 需要看到 Beta 结果）=====
            beta_raw = ""
            if risk.need_beta:
                yield self._phase("BETA_REVIEW")
                beta_prompt = self._build_beta_prompt(working_message, alpha_draft)
                tp_beta = _time.monotonic()
                self._last_collected = ""
                async for hb in self._stream_with_stage_timeout(
                    self._collect_with_heartbeat(self.beta_engine, beta_prompt, all_stats, system=BETA_SYSTEM),
                    "Beta", BETA_STAGE_TIMEOUT_SECONDS
                ):
                    yield hb
                beta_raw = self._last_collected
                timings["BETA"] = round(_time.monotonic() - tp_beta, 2)
                if beta_raw.strip():
                    yield f"\n{beta_raw.strip()}\n"

            # ===== Gamma 终审（能看到 Beta 结果）=====
            gamma_raw = ""
            if risk.need_gamma:
                yield self._phase("GAMMA_FINAL_REVIEW")
                gamma_prompt = self._build_gamma_prompt(working_message, alpha_draft, beta_raw)
                tp_gamma = _time.monotonic()
                self._last_collected = ""
                async for hb in self._stream_with_stage_timeout(
                    self._collect_with_heartbeat(self.gamma_engine, gamma_prompt, all_stats, system=GAMMA_SYSTEM),
                    "Gamma", GAMMA_STAGE_TIMEOUT_SECONDS
                ):
                    yield hb
                gamma_raw = self._last_collected
                timings["GAMMA"] = round(_time.monotonic() - tp_gamma, 2)
                if gamma_raw.strip():
                    yield f"\n{gamma_raw.strip()}\n"

            # ===== 解析审查结果 =====
            needs_rewrite = False
            combined_feedback = ""

            # Beta 解析
            beta_json = self._parse_json_review(beta_raw)
            beta_fail = False
            if beta_json and isinstance(beta_json, dict):
                beta_fail = beta_json.get("verdict", "").upper() == "FAIL"
            elif beta_raw.strip():
                beta_fail = "FAIL" in beta_raw.upper()

            # Gamma 解析
            gamma_json = self._parse_json_review(gamma_raw)
            gamma_fail = False
            if gamma_json and isinstance(gamma_json, dict):
                gamma_fail = gamma_json.get("verdict", "").upper() == "FAIL"
                gc = gamma_json.get("confidence", 0.5)
                if isinstance(gc, (int, float)) and gc < self.confidence_threshold:
                    gamma_fail = True
            elif gamma_raw.strip():
                gamma_fail = "FAIL" in gamma_raw.upper()

            if beta_fail or gamma_fail:
                needs_rewrite = True
                parts = []
                if beta_fail:
                    if beta_json:
                        issues = beta_json.get("issues", [])
                        fix = beta_json.get("fix", "")
                        imps = beta_json.get("improvements", [])
                        line = f"[Beta审查]\n问题：{'; '.join(issues) if issues else '无'}\n修正：{fix}"
                        if imps:
                            line += f"\n改进建议：{'; '.join(imps)}"
                        parts.append(line)
                    else:
                        parts.append(f"[Beta审查]\n{beta_raw.strip()[:500]}")
                if gamma_fail:
                    if gamma_json:
                        issues = gamma_json.get("issues", [])
                        fix = gamma_json.get("fix", "")
                        imps = gamma_json.get("improvements", [])
                        line = f"[Gamma终审]\n问题：{'; '.join(issues) if issues else '无'}\n修正：{fix}"
                        if imps:
                            line += f"\n改进建议：{'; '.join(imps)}"
                        parts.append(line)
                    else:
                        parts.append(f"[Gamma终审]\n{gamma_raw.strip()[:500]}")
                combined_feedback = "\n\n".join(parts)

            # ===== Rewrite or output =====
            if needs_rewrite and combined_feedback.strip():
                yield self._phase("ALPHA_REWRITE")
                rp = self._build_alpha_rewrite_prompt(context_text, working_message, alpha_draft, combined_feedback)
                final_ans = ""
                tp = _time.monotonic()
                async for item in self._stream_with_stage_timeout(
                    self._with_heartbeat(self.alpha_engine.execute(prompt=rp, system=ALPHA_SYSTEM)),
                    "Alpha 改写", ALPHA_STAGE_TIMEOUT_SECONDS
                ):
                    is_hb, chunk = item
                    if is_hb: yield chunk
                    else:
                        c, s = self._strip_token_stats(chunk); all_stats.extend(s); final_ans += c; yield c
                timings["ALPHA_REWRITE"] = round(_time.monotonic() - tp, 2)
                result = self._clean_final_answer(final_ans) if final_ans.strip() else self._clean_final_answer(alpha_draft)
            else:
                result = self._clean_final_answer(alpha_draft)

            self.memory.save_message(chat_id, "AI", result)
            yield self._phase("FINAL"); yield result
            yield f"\n[[TRINITAS_STATS:{_json.dumps(self._build_stats(t0, timings, all_stats))}]]\n"

        except StageTimeoutError as e:
            self.memory.save_message(chat_id, "AI", f"[阶段超时] {e.stage_name} 未在 {e.limit_seconds} 秒内完成。")
            yield self._phase("FINAL")
            yield f"\n[[TRINITAS_STATS:{_json.dumps(self._build_stats(t0, {}, []))}]]\n"
        except Exception as e:
            yield f"\n\n[崩溃]: {type(e).__name__} - {e}\n{traceback.format_exc()}"
            self.memory.save_message(chat_id, "AI", "[崩溃]")
