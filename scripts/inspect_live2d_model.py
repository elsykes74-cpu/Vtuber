"""Inspect a Live2D model3.json file and print setup hints.

This helper is intentionally read-only. It does not edit model_dict.json,
character configs, or model files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ROOT = Path("live2d-models")


def _case_get(data: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in data:
        return data[key]
    lower_key = key.lower()
    for actual_key, value in data.items():
        if actual_key.lower() == lower_key:
            return value
    return default


def _resolve_model3_path(path: Path) -> Path:
    if path.is_file():
        return path

    matches = sorted(path.rglob("*.model3.json"))
    if not matches:
        raise FileNotFoundError(f"No *.model3.json file found under {path}")
    if len(matches) > 1:
        joined = "\n  - ".join(str(match) for match in matches)
        raise ValueError(
            f"Multiple *.model3.json files found under {path}. "
            f"Please pass one explicitly:\n  - {joined}"
        )
    return matches[0]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _relative_file_status(model_dir: Path, file_name: str | None) -> dict[str, Any]:
    if not file_name:
        return {"file": "", "exists": False}
    file_path = model_dir / file_name
    return {"file": file_name, "exists": file_path.exists()}


def _extract_expressions(
    file_references: dict[str, Any], model_dir: Path
) -> list[dict[str, Any]]:
    expressions = _case_get(file_references, "Expressions", [])
    if not isinstance(expressions, list):
        return []

    output = []
    for index, expression in enumerate(expressions):
        if not isinstance(expression, dict):
            continue
        name = str(_case_get(expression, "Name", ""))
        file_name = _case_get(expression, "File", "")
        output.append(
            {
                "index": index,
                "name": name,
                **_relative_file_status(model_dir, file_name),
            }
        )
    return output


def _extract_motions(
    file_references: dict[str, Any], model_dir: Path
) -> dict[str, list[dict[str, Any]]]:
    motions = _case_get(file_references, "Motions", {})
    if not isinstance(motions, dict):
        return {}

    output: dict[str, list[dict[str, Any]]] = {}
    for group_name, entries in motions.items():
        if not isinstance(entries, list):
            continue
        output[str(group_name)] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            file_name = _case_get(entry, "File", "")
            output[str(group_name)].append(_relative_file_status(model_dir, file_name))
    return output


def _extract_hit_areas(data: dict[str, Any]) -> list[dict[str, str]]:
    hit_areas = _case_get(data, "HitAreas", [])
    if not isinstance(hit_areas, list):
        return []

    output = []
    for hit_area in hit_areas:
        if not isinstance(hit_area, dict):
            continue
        output.append(
            {
                "id": str(_case_get(hit_area, "Id", "")),
                "name": str(_case_get(hit_area, "Name", "")),
            }
        )
    return output


def _suggest_model_url(model3_path: Path, model_root: Path) -> str:
    try:
        relative = model3_path.resolve().relative_to(model_root.resolve())
    except ValueError:
        return "<path-to-model3.json>"
    return "/" + str((Path("live2d-models") / relative).as_posix())


def _suggest_model_name(model3_path: Path, model_root: Path) -> str:
    try:
        relative = model3_path.resolve().relative_to(model_root.resolve())
    except ValueError:
        return model3_path.parent.name

    if relative.parts:
        return relative.parts[0]
    return model3_path.parent.name


def _find_idle_group(motions: dict[str, list[dict[str, Any]]]) -> str | None:
    for preferred in ("idle", "Idle"):
        if preferred in motions:
            return preferred
    for group_name in motions:
        if group_name.lower() == "idle":
            return group_name
    return next(iter(motions), None)


def inspect_model(
    model3_path: Path,
    model_root: Path,
    model_name: str | None,
    character_name: str | None,
) -> dict[str, Any]:
    resolved_model3_path = _resolve_model3_path(model3_path)
    data = _read_json(resolved_model3_path)
    model_dir = resolved_model3_path.parent
    file_references = _case_get(data, "FileReferences", {})
    if not isinstance(file_references, dict):
        file_references = {}

    motions = _extract_motions(file_references, model_dir)
    suggested_model_name = model_name or _suggest_model_name(
        resolved_model3_path, model_root
    )
    suggested_character_name = character_name or suggested_model_name.replace("-", " ")
    idle_group = _find_idle_group(motions)

    model_dict_entry: dict[str, Any] = {
        "name": suggested_model_name,
        "description": "",
        "url": _suggest_model_url(resolved_model3_path, model_root),
        "kScale": 0.5,
        "initialXshift": 0,
        "initialYshift": 0,
    }
    if idle_group is not None:
        model_dict_entry["idleMotionGroupName"] = idle_group

    return {
        "model3_path": str(resolved_model3_path),
        "version": _case_get(data, "Version", ""),
        "moc": _case_get(file_references, "Moc", ""),
        "texture_count": len(_case_get(file_references, "Textures", []) or []),
        "expressions": _extract_expressions(file_references, model_dir),
        "motions": motions,
        "hit_areas": _extract_hit_areas(data),
        "suggested_model_dict_entry": model_dict_entry,
        "suggested_character_config": {
            "conf_name": suggested_model_name,
            "conf_uid": f"{suggested_model_name}_001",
            "live2d_model_name": suggested_model_name,
            "character_name": suggested_character_name,
            "avatar": "",
        },
    }


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_None found._"

    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        " | ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def _format_text(summary: dict[str, Any]) -> str:
    expression_rows = [
        [
            str(expression["index"]),
            expression["name"],
            expression["file"],
            "yes" if expression["exists"] else "no",
        ]
        for expression in summary["expressions"]
    ]

    motion_rows = []
    for group_name, entries in summary["motions"].items():
        existing = sum(1 for entry in entries if entry["exists"])
        motion_rows.append([group_name or "<empty>", str(len(entries)), str(existing)])

    hit_area_rows = [
        [hit_area["id"], hit_area["name"]] for hit_area in summary["hit_areas"]
    ]

    character_config = summary["suggested_character_config"]
    character_yaml = "\n".join(
        [
            f"conf_name: '{character_config['conf_name']}'",
            f"conf_uid: '{character_config['conf_uid']}'",
            f"live2d_model_name: '{character_config['live2d_model_name']}'",
            f"character_name: '{character_config['character_name']}'",
            "avatar: ''",
            "",
            "persona_prompt: |",
            "  Describe this character's personality and speaking style here.",
        ]
    )

    return "\n\n".join(
        [
            "# Live2D model inspection",
            f"Model file: {summary['model3_path']}",
            f"Version: {summary['version']}",
            f"MOC: {summary['moc']}",
            f"Texture count: {summary['texture_count']}",
            "## Expressions",
            _format_table(["Index", "Name", "File", "Exists"], expression_rows),
            "## Motion groups",
            _format_table(["Group", "Total", "Existing files"], motion_rows),
            "## Hit areas",
            _format_table(["Id", "Name"], hit_area_rows),
            "## Minimal model_dict.json entry",
            "```json\n"
            + json.dumps(summary["suggested_model_dict_entry"], indent=2)
            + "\n```",
            "## Character config starter",
            "```yaml\n" + character_yaml + "\n```",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a Live2D *.model3.json file and print setup hints."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a *.model3.json file, or a model directory containing one.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Root folder for local Live2D models. Defaults to live2d-models.",
    )
    parser.add_argument(
        "--name",
        help="Model name to use in the suggested model_dict.json entry.",
    )
    parser.add_argument(
        "--character-name",
        help="Character name to use in the suggested character config.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the text report.",
    )
    args = parser.parse_args()

    summary = inspect_model(
        model3_path=args.path,
        model_root=args.model_root,
        model_name=args.name,
        character_name=args.character_name,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(_format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
