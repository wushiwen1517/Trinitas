import os
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class IntegrityCheck:
    rel_path: str
    must_contain: Tuple[str, ...]


DEFAULT_BACKUP_DIR = os.environ.get("TRINITAS_BACKUP_DIR") or r"D:\Trinitas_backup"

# 只在“明显回滚/缺关键能力”时触发修复，避免覆盖用户的正常修改
CHECKS: Tuple[IntegrityCheck, ...] = (
    IntegrityCheck(
        rel_path=r"core\orchestrator.py",
        must_contain=(
            "BETA_FEWSHOT",
            "GAMMA_FEWSHOT",
            "CONTEXT_BUILD",
            "_collect_with_heartbeat",
            "_clean_final_answer",
        ),
    ),
    IntegrityCheck(
        rel_path=r"core\protocol.py",
        must_contain=(
            "class OllamaStreamExecutor",
            "async def stream",
            "TOKEN_STATS",
            "keep_alive",
        ),
    ),
    IntegrityCheck(
        rel_path=r"web\\index.html",
        must_contain=(
            "STREAM_IDLE_TIMEOUT_MS",
            "model-dropdown",
            "modelTrigger",
            "cleanInternalTags",
            "PHASE_LABELS",
            "total_eval_tokens",
        ),
    ),
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _looks_reverted(root: str) -> List[str]:
    problems: List[str] = []
    for ck in CHECKS:
        abs_path = os.path.join(root, ck.rel_path)
        if not os.path.exists(abs_path):
            problems.append(f"missing:{ck.rel_path}")
            continue
        text = _read_text(abs_path)
        for needle in ck.must_contain:
            if needle not in text:
                problems.append(f"marker_missing:{ck.rel_path}:{needle}")
                break
    return problems


def _restore_from_backup(root: str, backup_dir: str) -> bool:
    ok = True
    for ck in CHECKS:
        src = os.path.join(backup_dir, ck.rel_path)
        dst = os.path.join(root, ck.rel_path)
        if not os.path.exists(src):
            ok = False
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return ok


def ensure_integrity(backup_dir: str = DEFAULT_BACKUP_DIR) -> None:
    """
    启动自检：检测到“关键能力缺失/明显回滚”时，自动尝试从备份目录恢复。
    - 不会因为失败而阻断服务启动
    - 仅在 markers 缺失时才触发恢复（尽量避免覆盖正常改动）
    """
    root = _repo_root()
    problems = _looks_reverted(root)
    if not problems:
        return

    # 记录问题，方便定位“为什么觉得又变简单了”
    try:
        log_path = os.path.join(root, "integrity_guard.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("DETECTED_REVERT:\n")
            for p in problems:
                f.write(f"- {p}\n")
            f.write(f"backup_dir={backup_dir}\n\n")
    except Exception:
        pass

    if backup_dir and os.path.isdir(backup_dir):
        _restore_from_backup(root, backup_dir)

