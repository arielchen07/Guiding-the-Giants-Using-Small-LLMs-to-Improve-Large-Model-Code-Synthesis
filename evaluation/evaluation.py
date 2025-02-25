import json
import ast
import operator
import math
from typing import List, Dict

with open("../data/human_eval_data_ambiguity_with_soln_new.json", "r") as f:
    data = json.load(f)

OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

def safe_eval(expr, is_binary: bool = False):
    if not isinstance(expr, str):
        return expr

    expr_stripped = expr.strip()

    if is_binary:
        if expr_stripped.lower().startswith('0b'):
            try:
                return int(expr_stripped, 2)
            except ValueError:
                pass
        if expr_stripped.lower().startswith('0x'):
            try:
                return int(expr_stripped, 16)
            except ValueError:
                pass
        if all(ch in '01' for ch in expr_stripped):
            try:
                return int(expr_stripped, 2)
            except ValueError:
                pass
    try:
        return ast.literal_eval(expr_stripped)
    except (ValueError, SyntaxError):
        try:
            return eval(expr_stripped, {"__builtins__": None}, {})
        except Exception:
            return expr_stripped


def run_tests(solution_code: str, function_name: str, tests: List[Dict], prompt: str) -> List[bool]:
    namespace = {}
    exec(solution_code, namespace)
    func = namespace.get(function_name)
    if not func or not callable(func):
        raise ValueError(f"Function {function_name} not found.")

    is_binary = "bitwise" in prompt.lower() or "binary" in prompt.lower()
    
        

    results = []
    for test in tests:
        input_str = test["input"]
        expected_str = test["output"]
        relation = test["relation"]

        try:
            if "integer" in prompt.lower():
                is_binary_temp = False
            else:
                is_binary_temp = is_binary
            parsed_input = safe_eval(input_str, is_binary_temp)
            expected = safe_eval(expected_str, is_binary)

            args = (parsed_input,) if not isinstance(parsed_input, tuple) else parsed_input

            result = func(*args)
            if isinstance(result, str) and isinstance(expected, int):
                if is_binary:
                    result = int(result, 2)
                else:
                    result=int(result)
            if isinstance(expected, tuple) and isinstance(result, str):
                result = eval("("+result+")")
            # hardcode
            if isinstance(expected, float) and isinstance(result, float) and function_name!="find_zero":
                passed = abs(result - expected) < 1e-4

            elif relation in OPERATORS:
                passed = OPERATORS[relation](result, expected)
            else:
                env = {
                    "input_args": parsed_input,
                    "result": result, 
                    "expected": expected,
                    "math": math,
                    **namespace
                }
                relation_eval = relation.replace("$input$", "input_args").replace("candidate", "result")

                try:
                    local_env = {}
                    exec(relation_eval, env, local_env)  
                    passed = local_env.get("relation_result", False)
                except SyntaxError as e:
                    print(f"Syntax error in relation evaluation: {e}")
                    passed = False
            results.append(passed)

        except Exception as e:
            print(f"[ERROR] {function_name} failed on input={input_str!r}: {e}")
            results.append(False)

    return results


def get_main_function_name(prompt: str) -> str:
    tree = ast.parse(prompt)
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return funcs[-1] if funcs else None

def evaluate_all():
    summary = {}
    counter = 0 
    
    for entry in data:
        try:
            func_name = get_main_function_name(entry["prompt"])
            solution = entry["solution"]
            tests = ast.literal_eval(entry["tests"])

            test_results = run_tests(solution, func_name, tests, entry["prompt"])
            summary[func_name] = {
                "total": len(test_results),
                "passed": sum(test_results),
                "failed": len(test_results) - sum(test_results)
            }
            if len(test_results) - sum(test_results) ==0 and sum(test_results)!=0:
                counter+=1
            else:
                print(len(test_results) - sum(test_results))
                print(func_name)
                
        except Exception as e:
            print(f"Error evaluating {entry.get('prompt', '')[:20]}...: {e}")
    success_rate = counter/len(data)
    # print(counter)
    # print(len(data))
    summary["success_rate"] = success_rate
    return summary


if __name__ == "__main__":
    results = evaluate_all()
    print("pass @ 1: ", results["success_rate"])
    # for func, stats in results.items():
    #     print(func,stats)
