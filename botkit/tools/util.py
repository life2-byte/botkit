"""Utility tool (1): calculate. Exact, safe arithmetic (no eval of arbitrary code)."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..security import Policy
from ..tool import Tool, tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> Any:
    if len(expression) > 500:
        raise ValueError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ValueError("not a valid arithmetic expression") from None

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int) and right > 0:
                if max(left.bit_length(), 1) * right > 10_000:
                    raise ValueError("number too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            if len(node.args) > 10:
                raise ValueError("too many arguments")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError("unsupported expression (numbers, + - * / // % **, parentheses, sqrt/abs/round/min/max/log/sin/cos/tan/floor/ceil, pi, e)")

    return ev(tree)


def make(policy: Policy, sandbox=None) -> list[Tool]:
    @tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression exactly, e.g. "(1250 * 1.18) / 12" or "sqrt(2) * pi". Use this instead of doing maths in your head.

        Args:
            expression: The arithmetic expression.
        """
        return str(safe_eval(expression))

    return [calculate]
