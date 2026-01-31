from llm_nature_indention_compensation.repair import compile_and_repair


def test_compile_and_repair_fixes_simple_indent_error():
    bad = (
        "def f():\n"
        "    if True:\n"
        "      return 1\n"
    )
    fixed, err = compile_and_repair(bad, max_iters=6)
    assert err is None
    compile(fixed, "<string>", "exec")


def test_compile_and_repair_leaves_valid_code_valid():
    good = (
        "def f():\n"
        "    return 1\n"
    )
    fixed, err = compile_and_repair(good, max_iters=2)
    assert err is None
    compile(fixed, "<string>", "exec")


def test_compile_and_repair_reports_unfixable_syntax():
    bad = (
        "def f(:\n"
        "    pass\n"
    )
    fixed, err = compile_and_repair(bad, max_iters=2)
    assert isinstance(fixed, str)
    assert err is not None
