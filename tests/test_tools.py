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
    result = sql_query("SELECT title FROM papers WHERE topic = 'Computer Vision'")
    assert "Deep Residual Learning for Image Recognition" in result
    assert "BERT: Pre-training of Deep Bidirectional Transformers" not in result


def test_sql_query_rejects_non_select_statements():
    result = sql_query("DROP TABLE papers")
    assert result.startswith("[ERROR]")


def test_web_search_returns_results_for_a_real_query():
    result = web_search("LangGraph agent framework")
    assert not result.startswith("[ERROR]")
    assert len(result) > 0