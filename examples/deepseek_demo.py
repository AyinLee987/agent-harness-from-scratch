"""Quick smoke test: DeepSeekLLM + ReAct agent with calculator tool."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast, operator
from dotenv import load_dotenv
load_dotenv()

from agent import DeepSeekLLM, ReActAgent, ToolRegistry, tool

_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Expression): return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.
    Args:
        expression: An arithmetic expression, e.g. '23 * 17' or '(12 + 8) * 5'.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"
    if result == int(result):
        return str(int(result))
    return str(result)


def main():
    print("=" * 60)
    print("DeepSeekLLM smoke test")
    print("=" * 60)

    llm = DeepSeekLLM()
    print(f"Model: {llm.model}")
    print(f"Base URL: {llm._client.base_url}")

    agent = ReActAgent(llm=llm, tools=ToolRegistry([calculator]), max_steps=5)

    tasks = [
        "What is 23 times 17?",
        "Calculate (12 + 8) * 5 then subtract 30.",
    ]

    for task in tasks:
        print(f"\n>>> TASK: {task}")
        result = agent.run(task)
        for step in result.trajectory:
            if step["action"]:
                print(f"    [step {step['index']}] {step['action']['name']}"
                      f"({step['action']['arguments']}) → {step['observation'][:80]}")
        print(f"    ANSWER: {result.answer}")
        print(f"    steps={result.steps} tokens={result.tokens} stop={result.stop_reason}")


if __name__ == "__main__":
    main()
