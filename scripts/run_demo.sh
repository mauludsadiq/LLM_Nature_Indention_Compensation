python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m llm_nature_indention_compensation.run       --prompt "Write a Python function stress_test() with nested if/for/with and return 0."       --model gpt2       --device cpu       --max-new-tokens 256       --temperature 0.8       --top-p 0.95       --seed 0       --indent-delta 4       --max-depth 20       --indent-score first       --out out/run_demo
python -m llm_nature_indention_compensation.verify out/run_demo
