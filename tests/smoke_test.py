"""Offline smoke test for configuration integrity.

Validates every config template and character file through the same code
path the server uses at startup and on character switching, and checks
that the EN and ZH default templates stay structurally in sync.

Needs only the config_manager dependencies (pydantic, pyyaml, loguru,
chardet) — no models, no network, no torch. Run from the repo root:

    uv run python tests/smoke_test.py

or with a minimal environment:

    pip install pydantic pyyaml loguru chardet
    PYTHONPATH=src python tests/smoke_test.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from open_llm_vtuber.config_manager.utils import read_yaml, validate_config  # noqa: E402

TEMPLATES_DIR = REPO_ROOT / "config_templates"
CHARACTERS_DIR = REPO_ROOT / "characters"
BASE_TEMPLATE = TEMPLATES_DIR / "conf.default.yaml"
ZH_TEMPLATE = TEMPLATES_DIR / "conf.ZH.default.yaml"

failures: list[str] = []


def check(label: str, fn) -> None:
    try:
        fn()
        print(f"  ok: {label}")
    except Exception as e:
        failures.append(f"{label}: {e}")
        print(f"FAIL: {label}\n      {e}")


def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Mirror of service_context.deep_merge (kept local so this test
    does not import the heavy service_context module)."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def key_tree(data, prefix: str = "") -> set[str]:
    """Flatten nested dict keys into dotted paths (dicts only)."""
    keys: set[str] = set()
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            keys.add(path)
            keys |= key_tree(v, path)
    return keys


def validate_template(path: Path) -> None:
    validate_config(read_yaml(str(path)))


def validate_character(path: Path, base_config: dict) -> None:
    alt = read_yaml(str(path)).get("character_config")
    if alt is None:
        raise ValueError("file has no 'character_config' section")
    merged = deep_merge(base_config["character_config"], alt)
    validate_config(
        {"system_config": base_config["system_config"], "character_config": merged}
    )


def check_template_sync() -> None:
    en_keys = key_tree(read_yaml(str(BASE_TEMPLATE)))
    zh_keys = key_tree(read_yaml(str(ZH_TEMPLATE)))
    only_en = sorted(en_keys - zh_keys)
    only_zh = sorted(zh_keys - en_keys)
    if only_en or only_zh:
        raise ValueError(
            f"templates out of sync; only in conf.default.yaml: {only_en or 'none'}; "
            f"only in conf.ZH.default.yaml: {only_zh or 'none'}"
        )


def main() -> int:
    print("== config templates ==")
    for template in sorted(TEMPLATES_DIR.glob("*.yaml")):
        check(
            f"validate {template.relative_to(REPO_ROOT)}",
            lambda t=template: validate_template(t),
        )

    print("== character configs (merged over base template) ==")
    base_config = read_yaml(str(BASE_TEMPLATE))
    for character in sorted(CHARACTERS_DIR.glob("*.yaml")):
        check(
            f"validate {character.relative_to(REPO_ROOT)}",
            lambda c=character: validate_character(c, base_config),
        )

    print("== EN/ZH template sync ==")
    check(
        "conf.default.yaml and conf.ZH.default.yaml have identical key structure",
        check_template_sync,
    )

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
