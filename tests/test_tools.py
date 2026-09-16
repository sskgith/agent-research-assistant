"""
Tests for individual tool functions — correctness of each tool in isolation,
independent of the agent's decision-making.
"""

from tools.calculator_tool import calculator
from tools.sql_query_tool import sql_query
from tools.web_search_tool import web_search


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"


def test_calculator_operator_precedence():
    assert calculator("12 * (4 + 3)") == "84"


def test_calculator_division_by_zero_returns_error():
    result = calculator("1 / 0")
    assert result.startswith("[ERROR]")


def test_calculator_unsupported_operator_returns_error():
    result = calculator("2 % 2")  # modulo not in the allow-list
    assert result.startswith("[ERROR]")


def test_sql_query_returns_matching_rows():
    result = sql_query("SELECT name FROM products WHERE price > 10")
    assert "Widget B" in result
    assert "Widget C" in result
    assert "Widget A" not in result


def test_sql_query_rejects_non_select_statements():
    result = sql_query("DROP TABLE products")
    assert result.startswith("[ERROR]")


def test_web_search_returns_string_containing_query():
    result = web_search("test query")
    assert "test query" in result