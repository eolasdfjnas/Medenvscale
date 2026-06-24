from __future__ import annotations

import ast
import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class PathSafetyResult:
    raw_path: str
    normalized_path: str
    safe: bool
    reason: str = ""
    has_parent_ref: bool = False
    normalizes_to_workdir: bool = False
    directory_artifact: bool = False


@dataclass(frozen=True)
class CodePathReference:
    path: str
    operation: str
    lineno: int


def analyze_relative_path(path: Any, *, artifact: bool = False) -> PathSafetyResult:
    raw = str(path or "").strip()
    if not raw:
        return PathSafetyResult(raw, "", False, "empty_path")
    normalized_input = raw.replace("\\", "/")
    if normalized_input.startswith("~"):
        return PathSafetyResult(raw, normalized_input, False, "home_path")
    pure = PurePosixPath(normalized_input)
    if pure.is_absolute():
        return PathSafetyResult(raw, normalized_input, False, "absolute_path")
    normalized = posixpath.normpath(normalized_input)
    parts = PurePosixPath(normalized_input).parts
    has_parent_ref = ".." in parts
    if normalized == ".." or normalized.startswith("../"):
        return PathSafetyResult(raw, normalized, False, "path_escapes_workdir", has_parent_ref=has_parent_ref)
    normalizes_to_workdir = normalized in {"", "."}
    directory_artifact = artifact and (normalizes_to_workdir or normalized_input.endswith("/"))
    if directory_artifact:
        return PathSafetyResult(
            raw,
            normalized,
            False,
            "directory_file_artifact_path",
            has_parent_ref=has_parent_ref,
            normalizes_to_workdir=normalizes_to_workdir,
            directory_artifact=True,
        )
    return PathSafetyResult(
        raw,
        normalized,
        True,
        has_parent_ref=has_parent_ref,
        normalizes_to_workdir=normalizes_to_workdir,
        directory_artifact=directory_artifact,
    )


def extract_code_path_references(code: str) -> list[CodePathReference]:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return []
    visitor = _CodePathReferenceVisitor()
    visitor.visit(tree)
    return visitor.references


class _CodePathReferenceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bindings: dict[str, set[str]] = {}
        self.references: list[CodePathReference] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        values = _literal_string_values(node.value)
        if values:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.bindings[target.id] = set(values)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        target_name = node.target.id if isinstance(node.target, ast.Name) else ""
        values = _literal_string_values(node.iter)
        previous: set[str] | None = None
        if target_name and values:
            previous = set(self.bindings.get(target_name, set()))
            self.bindings[target_name] = set(values)
        for child in node.body:
            self.visit(child)
        if target_name and values:
            if previous:
                self.bindings[target_name] = previous
            else:
                self.bindings.pop(target_name, None)
        for child in node.orelse:
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:
        operation = _path_operation(node)
        if operation:
            for path in self._paths_from_node(_path_arg_node(node, operation)):
                self.references.append(CodePathReference(path=path, operation=operation, lineno=getattr(node, "lineno", 0)))
        self.generic_visit(node)

    def _paths_from_node(self, node: ast.AST | None) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return set(self.bindings.get(node.id, set()))
        return set()


def _path_operation(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name) and node.func.id == "open":
        return "open"
    if isinstance(node.func, ast.Name) and node.func.id == "Path":
        return "path_ctor"
    if isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        if _is_os_path_call(node.func, {"makedirs", "mkdir"}):
            return attr
        if attr in {"write_text", "write_bytes", "mkdir", "open"} and isinstance(node.func.value, ast.Call):
            inner = node.func.value
            if isinstance(inner.func, ast.Name) and inner.func.id == "Path":
                return f"path_{attr}"
    return ""


def _path_arg_node(node: ast.Call, operation: str) -> ast.AST | None:
    if operation.startswith("path_") and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
        inner = node.func.value
        return inner.args[0] if inner.args else None
    return node.args[0] if node.args else None


def _is_os_path_call(func: ast.Attribute, names: set[str]) -> bool:
    if func.attr not in names:
        return False
    value = func.value
    return (
        isinstance(value, ast.Attribute)
        and value.attr == "path"
        and isinstance(value.value, ast.Name)
        and value.value.id == "os"
    ) or (isinstance(value, ast.Name) and value.id == "os")


def _literal_string_values(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values: set[str] = set()
        for item in node.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.add(item.value)
            else:
                return set()
        return values
    return set()
