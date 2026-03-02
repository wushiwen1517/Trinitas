# core/router.py

from dataclasses import dataclass, field
from typing import Dict, List
import re


@dataclass
class RiskDecision:
    """
    路由决策对象（供 orchestrator 使用）
    """
    level: int
    need_beta: bool
    need_gamma: bool
    need_vision: bool

    route_name: str = ""
    confidence: float = 0.0

    # 方便前端/日志调试
    reasons: List[str] = field(default_factory=list)
    strategy_tags: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)


class RiskRouter:
    """
    工业版风险路由器（复杂展开版）
    目标：
    - 将请求分级为 L0~L3
    - 强制决定是否启用 Beta/Gamma 审查
    - 提供可解释路由原因
    """

    def __init__(self):
        # -----------------------------
        # 权重配置（可调）
        # -----------------------------
        self.weights = {
            "code": 1.2,
            "math_logic": 1.1,
            "architecture": 1.6,
            "ops_config": 0.9,
            "metacognition": 1.0,
            "multi_constraint": 1.2,
            "trap_riddle": 1.1,
            "long_request": 0.8,
            "image": 0.5,
            "production_safety": 1.5,
        }

        # -----------------------------
        # 关键词库（中文 + 英文）
        # -----------------------------
        self.keyword_banks = {
            "code": [
                "python", "java", "cpp", "c++", "javascript", "typescript", "go", "rust",
                "代码", "函数", "类", "脚本", "编程", "实现", "接口", "api", "fastapi",
                "asyncio", "并发代码", "重构", "调试", "报错", "traceback", "exception",
                "sql", "sqlite", "正则", "算法实现", "数据结构", "单元测试", "测试用例",
                "uvicorn", "ollama", "playwright", "docker", "dockerfile"
            ],
            "math_logic": [
                "数学", "证明", "推理", "逻辑", "严谨", "反例", "归纳", "演绎", "命题",
                "最优", "复杂度", "概率", "统计", "方程", "几何", "离散", "代数",
                "prove", "proof", "reasoning", "logic", "counterexample", "induction",
                "complexity", "optimize", "theorem", "lemma"
            ],
            "architecture": [
                "架构", "生产级", "工业级", "高可靠", "高并发", "分布式", "微服务", "容灾",
                "幂等", "降级", "熔断", "限流", "一致性", "可扩展", "可维护", "可观测",
                "状态机", "调度器", "编排", "orchestrator", "pipeline", "state machine",
                "distributed", "fault tolerance", "resilience", "reliability", "scalability"
            ],
            "ops_config": [
                "配置", "步骤", "部署", "命令", "安装", "怎么做", "启动", "运行", "环境变量",
                "端口", "路径", "代理", "隧道", "cloudflare", "cloudflared", "域名",
                "cmd", "powershell", "shell", "bash", "linux", "windows", "rocm", "vulkan",
                "install", "setup", "deploy", "config", "command", "run", "port"
            ],
            "metacognition": [
                "元认知", "自我反思", "为什么会错", "推理过程", "思考链", "审查", "裁判",
                "反思", "校验", "复核", "自检", "你为什么", "你确定", "你发誓",
                "metacognition", "self-check", "self critique", "review", "judge"
            ],
            "production_safety": [
                "安全", "鉴权", "权限", "token", "api key", "密钥", "access", "防火墙",
                "公网暴露", "zero trust", "cloudflare access", "认证", "授权", "审计",
                "日志脱敏", "敏感信息", "rate limit", "waf"
            ],
            "trap_riddle": [
                "脑筋急转弯", "陷阱题", "偏置测试", "假设测试", "你确定吗", "经典题", "一次观测",
                "置换", "全局偏移", "单点故障", "riddle", "trick question", "bias test"
            ],
        }

        # 多约束模式（regex）
        self.multi_constraint_patterns = [
            r"至少.{0,12}同时",
            r"必须.{0,20}并且",
            r"既要.{0,20}又要",
            r"不能.{0,20}还要",
            r"要求[:：].{0,80}(并发|重试|结果|异常|返回)",
            r"支持.{0,15}并支持",
            r"one.*and.*another",  # 粗略英文多约束
        ]

        # 高风险架构/生产触发模式（regex）
        self.architecture_patterns = [
            r"(高并发|分布式|容灾|熔断|降级|幂等)",
            r"(生产级|工业级|企业级)",
            r"(异步任务调度器|状态机|编排器|orchestrator|pipeline)",
            r"(多模型).{0,10}(审查|路由|裁判)",
            r"(可靠性|可扩展|可观测|可维护)",
        ]

        # 数学/严谨推理触发模式
        self.logic_patterns = [
            r"(证明|反证|归纳|反例|严谨推理)",
            r"(最小|最大).{0,12}(改动|代价|复杂度)",
            r"(是否可解|不可解|必要条件|充分条件)",
        ]

        # 代码类模式
        self.code_patterns = [
            r"```[\s\S]*?```",
            r"Traceback \(most recent call last\)",
            r"(SyntaxError|ImportError|ModuleNotFoundError|TypeError|ValueError|KeyError|sqlite3\.)",
            r"(uvicorn|fastapi|ollama|sqlite|python\s+-m)",
        ]

        # 路由阈值（可调）
        self.thresholds = {
            "L3": 6.5,
            "L2": 3.6,
            "L1": 1.8,
        }

        # 强制审查策略开关
        self.force_gamma_on_code = True
        self.force_gamma_on_math = True
        self.force_beta_on_metacognition = True

    # =========================================================
    # 公共接口
    # =========================================================
    def analyze(self, message: str, has_image: bool = False) -> RiskDecision:
        text = self._normalize(message)

        # 1) 收集原始信号
        raw_signals = self._collect_signals(text=text, has_image=has_image)

        # 2) 加权评分
        weighted_scores = self._weight_signals(raw_signals)

        # 3) 总分与分级
        total_score = sum(weighted_scores.values())
        level = self._determine_level(total_score, raw_signals)

        # 4) 生成决策
        decision = self._build_decision(
            level=level,
            text=text,
            has_image=has_image,
            raw_signals=raw_signals,
            weighted_scores=weighted_scores,
            total_score=total_score
        )

        return decision

    # =========================================================
    # 内部实现：文本预处理
    # =========================================================
    def _normalize(self, text: str) -> str:
        if text is None:
            return ""
        t = str(text)
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.strip().lower()
        return t

    # =========================================================
    # 内部实现：信号提取
    # =========================================================
    def _collect_signals(self, text: str, has_image: bool) -> Dict[str, float]:
        signals = {
            "code": 0.0,
            "math_logic": 0.0,
            "architecture": 0.0,
            "ops_config": 0.0,
            "metacognition": 0.0,
            "multi_constraint": 0.0,
            "trap_riddle": 0.0,
            "long_request": 0.0,
            "image": 0.0,
            "production_safety": 0.0,
        }

        # ---------- 关键词计数 ----------
        for category, kws in self.keyword_banks.items():
            hit_count = 0
            for kw in kws:
                if kw.lower() in text:
                    hit_count += 1

            # 命中数映射到 0~3 区间（防止无限膨胀）
            if hit_count > 0:
                if hit_count == 1:
                    signals[category] += 1.0
                elif hit_count == 2:
                    signals[category] += 1.8
                elif hit_count == 3:
                    signals[category] += 2.4
                else:
                    signals[category] += 3.0

        # ---------- 正则模式增强 ----------
        for p in self.architecture_patterns:
            if re.search(p, text, re.I):
                signals["architecture"] += 1.3

        for p in self.logic_patterns:
            if re.search(p, text, re.I):
                signals["math_logic"] += 1.2

        for p in self.code_patterns:
            if re.search(p, text, re.I):
                signals["code"] += 1.2

        for p in self.multi_constraint_patterns:
            if re.search(p, text, re.I):
                signals["multi_constraint"] += 1.0

        # ---------- 文本长度信号 ----------
        # 长请求通常更复杂，但不直接等于高风险
        char_len = len(text)
        if char_len >= 400:
            signals["long_request"] += 1.0
        if char_len >= 1000:
            signals["long_request"] += 1.2
        if char_len >= 2500:
            signals["long_request"] += 1.4

        # ---------- 图片信号 ----------
        if has_image:
            signals["image"] += 1.0

        # ---------- 组合触发增强 ----------
        # 代码 + 架构 + 安全 -> L3 倾向
        if signals["code"] > 0 and signals["architecture"] > 0 and signals["production_safety"] > 0:
            signals["architecture"] += 1.5

        # 数学逻辑 + 陷阱题 -> L2 倾向
        if signals["math_logic"] > 0 and signals["trap_riddle"] > 0:
            signals["math_logic"] += 1.2

        # 元认知 + 审查 -> 审查链倾向
        if signals["metacognition"] > 0 and ("审查" in text or "复核" in text or "judge" in text):
            signals["metacognition"] += 1.0

        return signals

    # =========================================================
    # 内部实现：加权
    # =========================================================
    def _weight_signals(self, raw_signals: Dict[str, float]) -> Dict[str, float]:
        weighted = {}
        for k, v in raw_signals.items():
            weighted[k] = round(v * self.weights.get(k, 1.0), 3)
        return weighted

    # =========================================================
    # 内部实现：等级判定
    # =========================================================
    def _determine_level(self, total_score: float, raw_signals: Dict[str, float]) -> int:
        # 强触发规则（优先于总分）
        if raw_signals["architecture"] >= 2.4:
            return 3
        if raw_signals["production_safety"] >= 2.0 and raw_signals["code"] >= 1.0:
            return 3

        if raw_signals["math_logic"] >= 2.0:
            return 2
        if raw_signals["trap_riddle"] >= 1.5:
            return 2

        # 总分阈值
        if total_score >= self.thresholds["L3"]:
            return 3
        if total_score >= self.thresholds["L2"]:
            return 2
        if total_score >= self.thresholds["L1"]:
            return 1
        return 0

    # =========================================================
    # 内部实现：构建最终决策
    # =========================================================
    def _build_decision(
        self,
        level: int,
        text: str,
        has_image: bool,
        raw_signals: Dict[str, float],
        weighted_scores: Dict[str, float],
        total_score: float
    ) -> RiskDecision:
        reasons: List[str] = []
        strategy_tags: List[str] = []

        # 路由名称
        route_name = f"L{level}"

        # 是否启用视觉
        need_vision = bool(has_image)

        # 基础规则：强制审查
        need_beta = False
        need_gamma = False

        if level == 0:
            need_beta = False
            need_gamma = False
            strategy_tags.append("fast-path")

        elif level == 1:
            # 配置/步骤类：默认启用 Gamma 终审（强制审查策略）
            need_beta = False
            need_gamma = True
            strategy_tags.append("ops-review")
            strategy_tags.append("gamma-only")

        elif level == 2:
            # 数学/逻辑/多约束：强制 Beta + Gamma
            need_beta = True
            need_gamma = True
            strategy_tags.append("logic-chain")
            strategy_tags.append("forced-review")

        elif level == 3:
            # 架构/生产级：强制 Beta + Gamma，且记高风险标签
            need_beta = True
            need_gamma = True
            strategy_tags.append("architecture-chain")
            strategy_tags.append("forced-review")
            strategy_tags.append("high-rigor")

        # 细化覆盖规则（强制审查策略）
        if self.force_gamma_on_code and raw_signals["code"] >= 1.8:
            need_gamma = True
            strategy_tags.append("code-gamma-enforced")

        if self.force_gamma_on_math and raw_signals["math_logic"] >= 1.4:
            need_gamma = True
            strategy_tags.append("math-gamma-enforced")

        if self.force_beta_on_metacognition and raw_signals["metacognition"] >= 1.2:
            need_beta = True
            strategy_tags.append("meta-beta-enforced")

        # 图片 + 文本复杂逻辑 -> 走完整链
        if has_image and (raw_signals["math_logic"] > 0 or raw_signals["code"] > 0 or raw_signals["architecture"] > 0):
            need_beta = True
            need_gamma = True
            strategy_tags.append("vision-plus-reasoning")

        # 可解释 reasons
        reasons.extend(self._generate_reasons(raw_signals, total_score, level, has_image))

        # 简单置信度（路由自身，不是答案置信度）
        router_conf = self._estimate_router_confidence(raw_signals, total_score, level)

        return RiskDecision(
            level=level,
            need_beta=need_beta,
            need_gamma=need_gamma,
            need_vision=need_vision,
            route_name=route_name,
            confidence=router_conf,
            reasons=reasons,
            strategy_tags=self._dedupe_keep_order(strategy_tags),
            scores={
                "total": round(total_score, 3),
                **weighted_scores
            }
        )

    # =========================================================
    # 辅助：理由生成
    # =========================================================
    def _generate_reasons(
        self,
        raw_signals: Dict[str, float],
        total_score: float,
        level: int,
        has_image: bool
    ) -> List[str]:
        reasons = []

        if has_image:
            reasons.append("检测到图片输入，需要视觉链路")

        if raw_signals["architecture"] >= 1.5:
            reasons.append("命中架构/生产级关键词或模式，提升到高严谨路由")

        if raw_signals["math_logic"] >= 1.2:
            reasons.append("命中数学/逻辑/严谨推理信号，建议启用审查链")

        if raw_signals["code"] >= 1.2:
            reasons.append("命中代码/报错/Traceback 信号，建议代码审查")

        if raw_signals["ops_config"] >= 1.0:
            reasons.append("命中配置/部署/命令类问题，至少进行终审检查")

        if raw_signals["metacognition"] >= 1.0:
            reasons.append("命中元认知/复核要求，启用额外审查层")

        if raw_signals["multi_constraint"] >= 1.0:
            reasons.append("检测到多约束表达，复杂度上升")

        if raw_signals["trap_riddle"] >= 1.0:
            reasons.append("检测到陷阱题/偏置测试倾向，需更谨慎")

        reasons.append(f"总路由评分={round(total_score, 3)}，判定 L{level}")

        return reasons

    # =========================================================
    # 辅助：路由置信度估算（不是答案置信度）
    # =========================================================
    def _estimate_router_confidence(
        self,
        raw_signals: Dict[str, float],
        total_score: float,
        level: int
    ) -> float:
        # 经验规则：强信号越多、总分越高，路由更稳定
        strong_signal_count = sum(1 for v in raw_signals.values() if v >= 1.0)

        base = 0.55
        base += min(total_score / 12.0, 0.20)
        base += min(strong_signal_count * 0.03, 0.15)

        # L3/L2 的高强触发通常更稳定
        if level >= 2 and (raw_signals["architecture"] >= 1.5 or raw_signals["math_logic"] >= 1.2):
            base += 0.06

        # 只有长度信号而没其他特征时，置信度下降
        if raw_signals["long_request"] > 0 and strong_signal_count <= 1:
            base -= 0.08

        base = max(0.0, min(0.99, base))
        return round(base, 3)

    # =========================================================
    # 辅助：去重
    # =========================================================
    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out