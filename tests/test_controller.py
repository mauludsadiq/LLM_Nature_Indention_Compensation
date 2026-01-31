from llm_nature_indention_compensation.indent_controller import IndentController


def test_indent_stack_push_pop():
    c = IndentController.init(indent_delta=4)
    assert c.indent_stack == [0]
    c.prev_line_ended_with_colon = True
    assert c.allowed_indents("x") == [4]
    c.update_stack(4)
    assert c.indent_stack == [0, 4]
    c.prev_line_ended_with_colon = False
    assert set(c.allowed_indents("x")) == {0, 4}
    c.update_stack(0)
    assert c.indent_stack == [0]
