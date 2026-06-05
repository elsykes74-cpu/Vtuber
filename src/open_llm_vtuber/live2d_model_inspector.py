import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse


MODEL3_REQUIRED_KEYS = ("Version", "FileReferences")
MODEL_DICT_REQUIRED_KEYS = ("name", "url", "emotionMap")
RESOURCE_FILE_KEYS = (
    "Moc",
    "Physics",
    "Pose",
    "DisplayInfo",
    "UserData",
)
MOTION_KEYS = ("Motions", "Idle", "TapBody", "TapHead")
EXPRESSION_KEYS = ("Expressions",)
TEXTURE_KEYS = ("Textures",)
LOCAL_STATIC_PREFIXES = ("/live2d-models/", "live2d-models/")
REMOTE_SCHEMES = ("http", "https", "ws", "wss")


@dataclass
class InspectionIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.field:
            payload["field"] = self.field
        return payload


@dataclass
class ResourceRef:
    category: str
    path: str
    source: str
    exists: bool | None = None
    absolute_path: str | None = None
    group: str | None = None
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "path": self.path,
            "source": self.source,
            "exists": self.exists,
        }
        if self.absolute_path:
            payload["absolute_path"] = self.absolute_path
        if self.group:
            payload["group"] = self.group
        if self.index is not None:
            payload["index"] = self.index
        return payload


@dataclass
class MotionGroupSummary:
    name: str
    count: int = 0
    missing_count: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, entry: dict[str, Any], exists: bool | None) -> None:
        self.count += 1
        if exists is False:
            self.missing_count += 1
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "missing_count": self.missing_count,
            "entries": self.entries,
        }


class Live2DModelInspector:
    """Inspect Live2D model dictionary entries and Cubism model3 files."""

    def __init__(
        self,
        project_root: str = ".",
        live2d_models_dir: str = "live2d-models",
        model_dict_path: str = "model_dict.json",
    ):
        self.project_root = os.path.abspath(project_root)
        self.live2d_models_dir = live2d_models_dir
        self.model_dict_path = model_dict_path

    def inspect_model_dict(self, model_dict_path: str | None = None) -> dict[str, Any]:
        path = model_dict_path or self.model_dict_path
        absolute_path = self._resolve_project_path(path)
        issues: list[InspectionIssue] = []
        entries: list[dict[str, Any]] = []

        data = self._load_json_file(absolute_path, issues, source="model_dict")
        if data is None:
            return self._build_result(
                source_path=absolute_path,
                entries=[],
                resources=[],
                issues=issues,
                extra={"model_count": 0, "valid_model_count": 0},
            )

        if not isinstance(data, list):
            issues.append(
                InspectionIssue(
                    severity="error",
                    code="model-dict-not-list",
                    message="model_dict.json should contain a list of model entries.",
                    path=absolute_path,
                )
            )
            return self._build_result(
                source_path=absolute_path,
                entries=[],
                resources=[],
                issues=issues,
                extra={"model_count": 0, "valid_model_count": 0},
            )

        names: set[str] = set()
        duplicate_names: set[str] = set()

        for index, entry in enumerate(data):
            entry_issues: list[InspectionIssue] = []
            if not isinstance(entry, dict):
                issues.append(
                    InspectionIssue(
                        severity="error",
                        code="model-entry-not-object",
                        message=f"Model entry at index {index} should be an object.",
                        path=absolute_path,
                        field=str(index),
                    )
                )
                continue

            name = str(entry.get("name", "")).strip()
            if name:
                if name in names:
                    duplicate_names.add(name)
                names.add(name)

            inspected = self.inspect_model_entry(entry, model_index=index)
            entry_issues.extend(
                InspectionIssue(
                    severity=item["severity"],
                    code=item["code"],
                    message=item["message"],
                    path=item.get("path"),
                    field=item.get("field"),
                )
                for item in inspected["issues"]
            )
            entries.append(
                {
                    "index": index,
                    "name": name,
                    "url": entry.get("url"),
                    "health": inspected["health"],
                    "summary": inspected["summary"],
                    "issues": [issue.to_dict() for issue in entry_issues],
                }
            )
            issues.extend(entry_issues)

        for name in sorted(duplicate_names):
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="duplicate-model-name",
                    message=f'Model name "{name}" appears more than once.',
                    path=absolute_path,
                    field="name",
                )
            )

        return self._build_result(
            source_path=absolute_path,
            entries=entries,
            resources=[],
            issues=issues,
            extra={
                "model_count": len(data),
                "valid_model_count": len([entry for entry in entries if entry["health"] != "error"]),
                "duplicate_model_names": sorted(duplicate_names),
            },
        )

    def inspect_model_entry(
        self, model_info: dict[str, Any], model_index: int | None = None
    ) -> dict[str, Any]:
        issues: list[InspectionIssue] = []
        resources: list[ResourceRef] = []

        self._validate_model_dict_entry(model_info, issues, model_index)
        model_url = str(model_info.get("url", "")).strip()
        model_path = self.resolve_live2d_path(model_url)

        if not model_path:
            issues.append(
                InspectionIssue(
                    severity="error",
                    code="missing-model-url",
                    message="Live2D model entry does not define a local model3 url.",
                    field="url",
                )
            )
            return self._build_result(
                source_path=model_url,
                entries=[],
                resources=[],
                issues=issues,
                extra=self._build_entry_summary(model_info, None, resources, issues),
            )

        model_exists = os.path.isfile(model_path)
        resources.append(
            ResourceRef(
                category="model3",
                path=model_url,
                source="model_dict.url",
                exists=model_exists,
                absolute_path=model_path,
            )
        )

        if not model_exists:
            issues.append(
                InspectionIssue(
                    severity="error",
                    code="model3-file-missing",
                    message=f'Live2D model3 file "{model_url}" does not exist.',
                    path=model_path,
                    field="url",
                )
            )
            return self._build_result(
                source_path=model_path,
                entries=[],
                resources=resources,
                issues=issues,
                extra=self._build_entry_summary(model_info, None, resources, issues),
            )

        model3_result = self.inspect_model_file(model_path)
        resources.extend(
            ResourceRef(
                category=item["category"],
                path=item["path"],
                source=item["source"],
                exists=item.get("exists"),
                absolute_path=item.get("absolute_path"),
                group=item.get("group"),
                index=item.get("index"),
            )
            for item in model3_result["resources"]
        )
        issues.extend(
            InspectionIssue(
                severity=item["severity"],
                code=item["code"],
                message=item["message"],
                path=item.get("path"),
                field=item.get("field"),
            )
            for item in model3_result["issues"]
        )

        extra = self._build_entry_summary(
            model_info,
            model3_result,
            resources,
            issues,
        )
        return self._build_result(
            source_path=model_path,
            entries=[],
            resources=resources,
            issues=issues,
            extra=extra,
        )

    def inspect_model_file(self, model3_path: str) -> dict[str, Any]:
        absolute_path = self._resolve_project_path(model3_path)
        issues: list[InspectionIssue] = []
        resources: list[ResourceRef] = []
        model3 = self._load_json_file(absolute_path, issues, source="model3")
        model_dir = os.path.dirname(absolute_path)

        if model3 is None:
            return self._build_result(
                source_path=absolute_path,
                entries=[],
                resources=[],
                issues=issues,
                extra={"motions": {}, "expressions": [], "textures": []},
            )

        if not isinstance(model3, dict):
            issues.append(
                InspectionIssue(
                    severity="error",
                    code="model3-not-object",
                    message="Cubism model3 JSON should be an object.",
                    path=absolute_path,
                )
            )
            return self._build_result(
                source_path=absolute_path,
                entries=[],
                resources=[],
                issues=issues,
                extra={"motions": {}, "expressions": [], "textures": []},
            )

        self._validate_model3_shape(model3, issues, absolute_path)
        file_refs = model3.get("FileReferences", {})
        if not isinstance(file_refs, dict):
            file_refs = {}

        self._collect_resource_file_refs(file_refs, model_dir, resources, issues)
        motion_summary = self._collect_motion_refs(file_refs, model_dir, resources, issues)
        expression_summary = self._collect_expression_refs(
            file_refs, model_dir, resources, issues
        )
        texture_summary = self._collect_texture_refs(file_refs, model_dir, resources, issues)

        return self._build_result(
            source_path=absolute_path,
            entries=[],
            resources=resources,
            issues=issues,
            extra={
                "motions": {key: value.to_dict() for key, value in motion_summary.items()},
                "expressions": expression_summary,
                "textures": texture_summary,
            },
        )

    def resolve_live2d_path(self, url_or_path: str) -> str | None:
        if not url_or_path:
            return None

        parsed = urlparse(url_or_path)
        if parsed.scheme in REMOTE_SCHEMES:
            return None

        normalized = unquote(parsed.path or url_or_path).replace("\\", "/")
        for prefix in LOCAL_STATIC_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                return os.path.join(self.project_root, self.live2d_models_dir, normalized)

        if normalized.startswith("/"):
            normalized = normalized.lstrip("/")

        return self._resolve_project_path(normalized)

    def summarize_model_entry(self, model_info: dict[str, Any]) -> dict[str, Any]:
        result = self.inspect_model_entry(model_info)
        return {
            "health": result["health"],
            "summary": result["summary"],
            "issues": result["issues"],
        }

    def _validate_model_dict_entry(
        self,
        entry: dict[str, Any],
        issues: list[InspectionIssue],
        index: int | None,
    ) -> None:
        field_prefix = str(index) if index is not None else None
        for key in MODEL_DICT_REQUIRED_KEYS:
            if key not in entry:
                issues.append(
                    InspectionIssue(
                        severity="error",
                        code="missing-model-dict-field",
                        message=f'Model dictionary entry is missing "{key}".',
                        field=self._join_field(field_prefix, key),
                    )
                )

        if "emotionMap" in entry and not isinstance(entry.get("emotionMap"), dict):
            issues.append(
                InspectionIssue(
                    severity="error",
                    code="invalid-emotion-map",
                    message="emotionMap should be an object mapping emotion tags to expression indices.",
                    field=self._join_field(field_prefix, "emotionMap"),
                )
            )
        elif isinstance(entry.get("emotionMap"), dict):
            self._validate_emotion_map(entry["emotionMap"], issues, field_prefix)

        if "tapMotions" in entry and not isinstance(entry.get("tapMotions"), dict):
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="invalid-tap-motions",
                    message="tapMotions should be an object keyed by Live2D hit area.",
                    field=self._join_field(field_prefix, "tapMotions"),
                )
            )

        for numeric_key in ("kScale", "initialXshift", "initialYshift", "kXOffset"):
            if numeric_key in entry and not isinstance(entry[numeric_key], (int, float)):
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="invalid-live2d-number",
                        message=f"{numeric_key} should be numeric.",
                        field=self._join_field(field_prefix, numeric_key),
                    )
                )

    def _validate_emotion_map(
        self,
        emotion_map: dict[str, Any],
        issues: list[InspectionIssue],
        field_prefix: str | None,
    ) -> None:
        normalized_names: set[str] = set()
        expression_indices: dict[int, list[str]] = {}

        for name, index in emotion_map.items():
            normalized = str(name).strip().lower()
            if not normalized:
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="blank-emotion-name",
                        message="emotionMap contains a blank emotion name.",
                        field=self._join_field(field_prefix, "emotionMap"),
                    )
                )
                continue
            if normalized in normalized_names:
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="duplicate-emotion-alias",
                        message=f'Emotion alias "{name}" duplicates another alias after lower-casing.',
                        field=self._join_field(field_prefix, f"emotionMap.{name}"),
                    )
                )
            normalized_names.add(normalized)

            if not isinstance(index, int):
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="emotion-index-not-int",
                        message=f'Emotion "{name}" should map to an integer expression index.',
                        field=self._join_field(field_prefix, f"emotionMap.{name}"),
                    )
                )
                continue
            expression_indices.setdefault(index, []).append(str(name))

        for index, aliases in expression_indices.items():
            if len(aliases) > 3:
                issues.append(
                    InspectionIssue(
                        severity="info",
                        code="many-emotions-share-expression",
                        message=f"Expression index {index} is reused by {len(aliases)} emotion aliases.",
                        field=self._join_field(field_prefix, "emotionMap"),
                    )
                )

    def _validate_model3_shape(
        self,
        model3: dict[str, Any],
        issues: list[InspectionIssue],
        model3_path: str,
    ) -> None:
        for key in MODEL3_REQUIRED_KEYS:
            if key not in model3:
                issues.append(
                    InspectionIssue(
                        severity="error",
                        code="missing-model3-field",
                        message=f'Cubism model3 JSON is missing "{key}".',
                        path=model3_path,
                        field=key,
                    )
                )

        file_refs = model3.get("FileReferences")
        if file_refs is not None and not isinstance(file_refs, dict):
            issues.append(
                InspectionIssue(
                    severity="error",
                    code="invalid-file-references",
                    message="FileReferences should be an object.",
                    path=model3_path,
                    field="FileReferences",
                )
            )

    def _collect_resource_file_refs(
        self,
        file_refs: dict[str, Any],
        model_dir: str,
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
    ) -> None:
        for key in RESOURCE_FILE_KEYS:
            value = file_refs.get(key)
            if not value:
                continue
            if not isinstance(value, str):
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="invalid-resource-reference",
                        message=f"{key} should be a file path string.",
                        field=f"FileReferences.{key}",
                    )
                )
                continue
            self._add_local_resource(
                resources=resources,
                issues=issues,
                model_dir=model_dir,
                category=key.lower(),
                path=value,
                source=f"FileReferences.{key}",
            )

    def _collect_motion_refs(
        self,
        file_refs: dict[str, Any],
        model_dir: str,
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
    ) -> dict[str, MotionGroupSummary]:
        motions = file_refs.get("Motions", {})
        summary: dict[str, MotionGroupSummary] = {}

        if not motions:
            return summary
        if not isinstance(motions, dict):
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="invalid-motions-reference",
                    message="FileReferences.Motions should be an object keyed by motion group.",
                    field="FileReferences.Motions",
                )
            )
            return summary

        for group_name, entries in motions.items():
            group_summary = MotionGroupSummary(name=str(group_name))
            summary[str(group_name)] = group_summary
            if not isinstance(entries, list):
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="invalid-motion-group",
                        message=f'Motion group "{group_name}" should be a list.',
                        field=f"FileReferences.Motions.{group_name}",
                    )
                )
                continue
            for index, motion in enumerate(entries):
                if not isinstance(motion, dict):
                    issues.append(
                        InspectionIssue(
                            severity="warning",
                            code="invalid-motion-entry",
                            message=f'Motion entry {index} in group "{group_name}" should be an object.',
                            field=f"FileReferences.Motions.{group_name}.{index}",
                        )
                    )
                    continue
                file_value = motion.get("File")
                if not isinstance(file_value, str) or not file_value:
                    issues.append(
                        InspectionIssue(
                            severity="warning",
                            code="missing-motion-file",
                            message=f'Motion entry {index} in group "{group_name}" is missing File.',
                            field=f"FileReferences.Motions.{group_name}.{index}.File",
                        )
                    )
                    continue
                ref = self._add_local_resource(
                    resources=resources,
                    issues=issues,
                    model_dir=model_dir,
                    category="motion",
                    path=file_value,
                    source="FileReferences.Motions",
                    group=str(group_name),
                    index=index,
                )
                group_summary.add(
                    {
                        "file": file_value,
                        "sound": motion.get("Sound"),
                        "fade_in": motion.get("FadeInTime"),
                        "fade_out": motion.get("FadeOutTime"),
                        "exists": ref.exists,
                    },
                    ref.exists,
                )
        return summary

    def _collect_expression_refs(
        self,
        file_refs: dict[str, Any],
        model_dir: str,
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
    ) -> list[dict[str, Any]]:
        expressions = file_refs.get("Expressions", [])
        summary: list[dict[str, Any]] = []

        if not expressions:
            return summary
        if not isinstance(expressions, list):
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="invalid-expressions-reference",
                    message="FileReferences.Expressions should be a list.",
                    field="FileReferences.Expressions",
                )
            )
            return summary

        for index, expression in enumerate(expressions):
            if not isinstance(expression, dict):
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="invalid-expression-entry",
                        message=f"Expression entry {index} should be an object.",
                        field=f"FileReferences.Expressions.{index}",
                    )
                )
                continue
            file_value = expression.get("File")
            name = expression.get("Name") or f"expression_{index}"
            if not isinstance(file_value, str) or not file_value:
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="missing-expression-file",
                        message=f'Expression "{name}" is missing File.',
                        field=f"FileReferences.Expressions.{index}.File",
                    )
                )
                continue
            ref = self._add_local_resource(
                resources=resources,
                issues=issues,
                model_dir=model_dir,
                category="expression",
                path=file_value,
                source="FileReferences.Expressions",
                group=str(name),
                index=index,
            )
            summary.append(
                {
                    "name": name,
                    "file": file_value,
                    "exists": ref.exists,
                }
            )
        return summary

    def _collect_texture_refs(
        self,
        file_refs: dict[str, Any],
        model_dir: str,
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
    ) -> list[dict[str, Any]]:
        textures = file_refs.get("Textures", [])
        summary: list[dict[str, Any]] = []

        if not textures:
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="missing-textures",
                    message="FileReferences.Textures is empty or missing.",
                    field="FileReferences.Textures",
                )
            )
            return summary
        if not isinstance(textures, list):
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="invalid-textures-reference",
                    message="FileReferences.Textures should be a list.",
                    field="FileReferences.Textures",
                )
            )
            return summary

        for index, texture in enumerate(textures):
            if not isinstance(texture, str) or not texture:
                issues.append(
                    InspectionIssue(
                        severity="warning",
                        code="invalid-texture-entry",
                        message=f"Texture entry {index} should be a file path string.",
                        field=f"FileReferences.Textures.{index}",
                    )
                )
                continue
            ref = self._add_local_resource(
                resources=resources,
                issues=issues,
                model_dir=model_dir,
                category="texture",
                path=texture,
                source="FileReferences.Textures",
                index=index,
            )
            summary.append(
                {
                    "file": texture,
                    "exists": ref.exists,
                }
            )
        return summary

    def _add_local_resource(
        self,
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
        model_dir: str,
        category: str,
        path: str,
        source: str,
        group: str | None = None,
        index: int | None = None,
    ) -> ResourceRef:
        absolute_path = os.path.normpath(os.path.join(model_dir, path))
        exists = os.path.isfile(absolute_path)
        ref = ResourceRef(
            category=category,
            path=path,
            source=source,
            exists=exists,
            absolute_path=absolute_path,
            group=group,
            index=index,
        )
        resources.append(ref)
        if not exists:
            issues.append(
                InspectionIssue(
                    severity="warning",
                    code="resource-file-missing",
                    message=f'{category} resource "{path}" does not exist.',
                    path=absolute_path,
                    field=source,
                )
            )
        return ref

    def _build_entry_summary(
        self,
        model_info: dict[str, Any],
        model3_result: dict[str, Any] | None,
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
    ) -> dict[str, Any]:
        emotion_map = model_info.get("emotionMap", {})
        if not isinstance(emotion_map, dict):
            emotion_map = {}

        expression_count = 0
        motion_group_count = 0
        motion_count = 0
        texture_count = 0
        if model3_result:
            expression_count = len(model3_result["summary"].get("expressions", []))
            motions = model3_result["summary"].get("motions", {})
            motion_group_count = len(motions)
            motion_count = sum(group.get("count", 0) for group in motions.values())
            texture_count = len(model3_result["summary"].get("textures", []))

        return {
            "name": model_info.get("name"),
            "url": model_info.get("url"),
            "emotion_count": len(emotion_map),
            "expression_count": expression_count,
            "motion_group_count": motion_group_count,
            "motion_count": motion_count,
            "texture_count": texture_count,
            "resource_count": len(resources),
            "missing_resource_count": len([item for item in resources if item.exists is False]),
            "warning_count": len([item for item in issues if item.severity == "warning"]),
            "error_count": len([item for item in issues if item.severity == "error"]),
        }

    def _build_result(
        self,
        source_path: str,
        entries: list[dict[str, Any]],
        resources: list[ResourceRef],
        issues: list[InspectionIssue],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        health = self._calculate_health(issues)
        resource_payload = [resource.to_dict() for resource in resources]
        issue_payload = [issue.to_dict() for issue in self._sort_issues(issues)]

        return {
            "source_path": source_path,
            "health": health,
            "summary": extra,
            "entries": entries,
            "resources": resource_payload,
            "issues": issue_payload,
        }

    def _calculate_health(self, issues: list[InspectionIssue]) -> str:
        if any(issue.severity == "error" for issue in issues):
            return "error"
        if any(issue.severity == "warning" for issue in issues):
            return "warning"
        return "ok"

    def _sort_issues(self, issues: list[InspectionIssue]) -> list[InspectionIssue]:
        weights = {"error": 0, "warning": 1, "info": 2}
        return sorted(
            issues,
            key=lambda issue: (
                weights.get(issue.severity, 9),
                issue.path or "",
                issue.field or "",
                issue.code,
            ),
        )

    def _load_json_file(
        self,
        path: str,
        issues: list[InspectionIssue],
        source: str,
    ) -> Any | None:
        if not os.path.isfile(path):
            issues.append(
                InspectionIssue(
                    severity="error",
                    code=f"{source}-file-missing",
                    message=f"JSON file does not exist: {path}",
                    path=path,
                )
            )
            return None

        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="utf-8-sig") as file:
                    return json.load(file)
            except Exception as exc:
                issues.append(
                    InspectionIssue(
                        severity="error",
                        code=f"{source}-json-read-error",
                        message=f"Unable to read JSON file: {exc}",
                        path=path,
                    )
                )
                return None
        except json.JSONDecodeError as exc:
            issues.append(
                InspectionIssue(
                    severity="error",
                    code=f"{source}-json-invalid",
                    message=f"JSON file is invalid: {exc}",
                    path=path,
                )
            )
            return None

    def _resolve_project_path(self, path: str) -> str:
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.project_root, path))

    def _join_field(self, prefix: str | None, field_name: str) -> str:
        if not prefix:
            return field_name
        return f"{prefix}.{field_name}"
