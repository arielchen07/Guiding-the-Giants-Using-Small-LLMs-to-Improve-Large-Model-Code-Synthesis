

import ast
import operator
import math
import signal
import json
from typing import List, Dict
from tqdm import tqdm

# Define timeout exception
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out.")


    
with open("codellama_output.json", "r") as f:
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
    errored = False
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
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)

            result = func(*args)

            signal.alarm(0)
            
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
            # if not passed:
            #     print("funcname", function_name)
            #     print("result", result)
            #     print("expected", expected)
            #     print("soln", solution_code)
            #     print("paresed in", input_str)
            #     print("paresed in", parsed_input)
            results.append(passed)

        except Exception as e:
            print(f"[ERROR] {function_name} failed on input={input_str!r}: {e}")
            results.append(False)
            errored = True 

    return results, errored


def get_main_function_name(prompt: str, solution_code: str) -> str:

   
    solution_tree = ast.parse(solution_code)
    solution_funcs = [n.name for n in ast.walk(solution_tree) if isinstance(n, ast.FunctionDef)]

    if len(solution_funcs) == 1:
        return solution_funcs[0]
    else:
        try:
            prompt_tree = ast.parse(prompt)
            prompt_funcs = [n.name for n in ast.walk(prompt_tree) if isinstance(n, ast.FunctionDef)]
            return prompt_funcs[-1]
        except Exception as e:
            return solution_funcs[0]


def evaluate_all():
    summary = {}
    correct_programs = 0  
    errored_programs = 0 
    wrong_output_programs = 0  
    correct_solution_idx = []

    
    for i in tqdm(range(len(data))):
        entry = data[i]
        
        try:
            func_name = get_main_function_name(entry["llm_prompt_filtered"].strip(" "), entry["solution"])
            solution = entry["solution"].strip(" ")
            tests = ast.literal_eval(entry["tests"])
        
            test_results, errored = run_tests(solution, func_name, tests, entry["llm_prompt_filtered"].strip(" "))
            passed_tests = sum(test_results)
            failed_tests = len(test_results) - passed_tests
            
            summary[func_name] = {
                "total": len(test_results),
                "passed": passed_tests,
                "failed": failed_tests,
                "errored": errored 
            }
           
            
            if passed_tests == len(test_results) and len(test_results) != 0:
                correct_programs += 1
                correct_solution_idx.append(i)
            elif errored:  
                errored_programs += 1
            else:  
                wrong_output_programs += 1
                
        except Exception as e:
            print(f"Error evaluating {entry.get('prompt', '')[:20]}...: {e}")
            errored_programs+=1
    total_programs = correct_programs + errored_programs + wrong_output_programs 
    success_rate = correct_programs / (total_programs)
    summary["correct_programs"] = correct_programs  
    summary["errored_programs"] = errored_programs  
    summary["wrong_output_programs"] = wrong_output_programs 
    summary["total_programs"] = total_programs  
    summary["success_rate"] = success_rate
    summary["correct_solutions"]=correct_solution_idx
    # print(correct_solution_idx)

    
    return summary


if __name__ == "__main__":
    results = evaluate_all()
    print("pass @ 1: ", results["success_rate"])
    print("Total programs:", results["total_programs"])  
    print("Correct programs:", results["correct_programs"]) 
    print("Errored programs:", results["errored_programs"])  
    print("Wrong output programs:", results["wrong_output_programs"]) 
    print("correct solution indexes", results["correct_solutions"]) 
