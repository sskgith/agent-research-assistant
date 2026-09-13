"""
Calculator tool.

Unlike the web search and SQL tools, this one is fully implemented rather
than stubbed — arithmetic evaluation doesn't need an external API, and it
gives us one "real" tool to exercise the graph loop against from the start.
Uses Python's ast module to evaluate expressions safely (no raw eval()).
"""

import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """
    Safely evaluate a basic arithmetic expression.

    Args:
        expression: A string like "12 * (4 + 3)".

    Returns:
        The result as a string, or an error message if the expression
        is invalid or uses unsupported operations.
    """
    try:
        parsed = ast.parse(expression, mode="eval").body
        result = _eval_node(parsed)
        return str(result)
    except Exception as exc:
        return f"[ERROR] Could not evaluate '{expression}': {exc}"
