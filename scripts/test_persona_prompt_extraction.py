#!/usr/bin/env python3
"""Test script: verify persona_prompt is fully extracted from conf.yaml."""

import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.open_llm_vtuber.config_manager.utils import read_yaml, validate_config


def run_test(config_path: Path, label: str) -> bool:
    """Run extraction test on a config file. Returns True if success."""
    print(f"\n--- 测试文件: {label} ({config_path.name}) ---\n")

    if not config_path.exists():
        print(f"  错误: 找不到 {config_path}")
        return False

    try:
        raw = read_yaml(str(config_path))
    except Exception as e:
        print(f"  [YAML 解析失败] {e}")
        print("\n  提示: persona_prompt 需使用 | 表示多行字面量，例如:")
        print("    persona_prompt: |")
        print("      [Role and Context]")
        print("      Your content here...")
        return False
    cc = raw.get("character_config", {})
    persona_raw = cc.get("persona_prompt")

    print(f"1. 原始 YAML 解析后 persona_prompt 类型: {type(persona_raw).__name__}")

    if persona_raw is None:
        print("   [失败] persona_prompt 为 None")
        return False

    persona_str = persona_raw if isinstance(persona_raw, str) else str(persona_raw)
    lines = persona_str.strip().split("\n")
    char_count = len(persona_str)
    line_count = len(lines)

    print(f"2. 字符数: {char_count}")
    print(f"3. 行数: {line_count}")

    # Key phrases to check (common in persona prompts)
    key_phrases = ["Markdown", "reply"]
    print("\n4. 关键短语检查:")
    for phrase in key_phrases:
        found = phrase in persona_str
        status = "✓" if found else "✗"
        print(f"   {status} {phrase!r}")

    # Pydantic validation (full config load)
    print("\n5. Pydantic 完整配置验证:")
    try:
        config = validate_config(raw)
        validated_persona = config.character_config.persona_prompt
        print(f"   验证通过, persona_prompt 长度: {len(validated_persona)} 字符")
    except Exception as e:
        print(f"   [失败] {e}")
        return False

    # Show first and last portions
    print("\n6. 内容预览 (前 300 字符):")
    print("-" * 40)
    print(persona_str[:300])
    if len(persona_str) > 300:
        print("...")
    print("-" * 40)

    print("\n7. 内容预览 (后 200 字符):")
    print("-" * 40)
    if len(persona_str) > 500:
        print("...")
        print(persona_str[-200:])
    else:
        print(persona_str[max(0, 300) :])
    print("-" * 40)

    ok = char_count > 50 and all(p in persona_str for p in key_phrases)
    if ok:
        print("\n  结论: persona_prompt 已完整提取 ✓")
    else:
        print("\n  结论: 部分关键短语缺失，可能未完整提取 ✗")
    return ok


def main() -> None:
    print("=== persona_prompt 提取测试 ===")

    # Test conf.yaml first
    ok1 = run_test(project_root / "conf.yaml", "用户配置")
    # Fallback: test template (usually has valid YAML)
    ok2 = run_test(project_root / "config_templates" / "conf.default.yaml", "英文模板")

    print("\n" + "=" * 40)
    if ok1:
        print("conf.yaml: 提取成功 ✓")
    else:
        print(
            "conf.yaml: 提取失败或 YAML 格式错误 (persona_prompt 需使用 persona_prompt: |)"
        )
    if ok2:
        print("conf.default.yaml: 提取成功 ✓")
    print("=" * 40)


if __name__ == "__main__":
    main()
