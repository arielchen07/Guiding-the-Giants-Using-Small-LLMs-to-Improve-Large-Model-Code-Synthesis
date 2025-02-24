import json
import re
import traceback

import ast



def execute_function(code, test_cases):
    try:
        exec_globals = {}  


        exec(code, exec_globals)  

        func_name = next(name for name in exec_globals if callable(exec_globals[name]))
        func = exec_globals[func_name]

        test_cases = json.loads(test_cases.replace("'", "\""))  
        print(test_cases)
        results = []

        for test in test_cases:
            try:
                args = ast.literal_eval(test['input'])  
                expected_output = ast.literal_eval(test['output']) 

                result = func(*args)
                passed = eval(f"{result} {test['relation']} {expected_output}")

                results.append({
                    'input': test['input'],
                    'expected_output': expected_output,
                    'actual_output': result,
                    'passed': passed
                })
            except Exception as e:
                results.append({
                    'input': test['input'],
                    'error': str(e),
                    'passed': False
                })

        return results
    except Exception as e:
        return [{'error': traceback.format_exc()}]

if __name__ == "__main__":
    #python_code = "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n"
    python_code = "\n\n\ndef has_close_elements(numbers, threshold):\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n"

    test_suite = "[{'input': '[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3', 'output': 'True', 'relation': '=='}, {'input': '[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05', 'output': 'False', 'relation': '=='}, {'input': '[1.0, 2.0, 5.9, 4.0, 5.0], 0.95', 'output': 'True', 'relation': '=='}, {'input': '[1.0, 2.0, 5.9, 4.0, 5.0], 0.8', 'output': 'False', 'relation': '=='}, {'input': '[1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1', 'output': 'True', 'relation': '=='}, {'input': '[1.1, 2.2, 3.1, 4.1, 5.1], 1.0', 'output': 'True', 'relation': '=='}, {'input': '[1.1, 2.2, 3.1, 4.1, 5.1], 0.5', 'output': 'False', 'relation': '=='}]"
    results = execute_function(python_code, test_suite)
    for result in results:
        print(result)
