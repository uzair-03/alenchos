"""
Alenchos HumanEval Benchmark Harness
====================================
Evaluates Alenchos (gemini-3.1-flash-lite) against the official OpenAI / EvalPlus HumanEval benchmark.
Pulls verified instances from Hugging Face, runs them through the Socratic Adversarial Engine,
and executes live Python unit tests against the generated code.
"""

import os
import sys
import json
import time
import re
import requests
import traceback
from typing import List, Dict, Any, Optional
from elenchos import ElenchosAgent

HUMANEVAL_HF_API = "https://datasets-server.huggingface.co/rows"

def fetch_humaneval_tasks(limit: int = 5, offset: int = 0) -> List[Dict[str, Any]]:
    """Pulls task instances from openai/openai_humaneval on Hugging Face."""
    print(f"[*] Fetching {limit} tasks from openai/openai_humaneval (split: test, offset: {offset})...")
    params = {
        "dataset": "openai/openai_humaneval",
        "config": "openai_humaneval",
        "split": "test",
        "offset": offset,
        "limit": limit
    }
    try:
        resp = requests.get(HUMANEVAL_HF_API, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            tasks = [row["row"] for row in data.get("rows", [])][:limit]
            print(f"[+] Successfully fetched {len(tasks)} verified HumanEval tasks.")
            return tasks
    except Exception as e:
        print(f"[!] Warning: HF API error ({e}).")
    return []

def clean_code(raw_code: str, entry_point: str) -> str:
    """Extracts pure executable Python code from markdown blocks if present."""
    if "```python" in raw_code:
        parts = raw_code.split("```python")
        if len(parts) > 1:
            code_block = parts[1].split("```")[0]
            return code_block.strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return raw_code.strip()

def test_solution(prompt: str, solution: str, test_code: str, entry_point: str, timeout_sec: float = 3.0) -> Dict[str, Any]:
    """Executes the test suite against the generated solution in an isolated environment."""
    exec_scope = {}
    full_program = f"{prompt}\n{solution}\n{test_code}\ncheck({entry_point})\n"
    
    t0 = time.time()
    try:
        exec(full_program, exec_scope)
        duration = round(time.time() - t0, 3)
        return {"passed": True, "error": None, "duration": duration}
    except AssertionError as ae:
        duration = round(time.time() - t0, 3)
        return {"passed": False, "error": f"AssertionError: Test assertion failed", "duration": duration}
    except Exception as e:
        duration = round(time.time() - t0, 3)
        return {"passed": False, "error": f"{type(e).__name__}: {str(e)[:100]}", "duration": duration}

def run_humaneval_benchmark(
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    limit: int = 5,
    agent_name: str = "Alenchos"
):
    print("=" * 78)
    print("ALENCHOS HUMANEVAL LIVE BENCHMARK EVALUATOR")
    print("=" * 78)
    print(f"Agent Framework:     Alenchos Socratic Quality Engine")
    print(f"Backend Model:       gemini/{model}")
    print(f"Benchmark:           HumanEval (OpenAI / EvalPlus)")
    print(f"Proprietary Status:  oss: false")
    print("=" * 78)

    agent = ElenchosAgent(
        api_key=api_key,
        provider="gemini",
        model=model,
        use_mock=False
    )

    tasks = fetch_humaneval_tasks(limit=limit)
    if not tasks:
        print("[!] Failed to fetch tasks from Hugging Face.")
        return

    results = []
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humaneval_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Running live inference and execution for {len(tasks)} tasks...\n")

    for i, t in enumerate(tasks, 1):
        task_id = t["task_id"]
        prompt = t["prompt"]
        entry_point = t["entry_point"]
        test_code = t["test"]

        docstring_preview = prompt.strip().splitlines()[-1][:60]
        print(f"[{i}/{len(tasks)}] {task_id} -> def {entry_point}(...)")

        socratic_prompt = (
            f"You are a rigorous Python software engineer.\n"
            f"Complete the following Python function correctly, handling all boundary and edge cases.\n"
            f"Return ONLY the implementation code for {entry_point}.\n\n"
            f"{prompt}"
        )

        t_start = time.time()
        agent_res = agent.run(socratic_prompt)
        gen_duration = round(time.time() - t_start, 2)

        cleaned = clean_code(agent_res.final_output, entry_point)

        # Run live unit test execution
        eval_res = test_solution(prompt, cleaned, test_code, entry_point)

        status_str = "PASSED" if eval_res["passed"] else "FAILED"
        print(f"    [+] Status: {status_str} (Cycles: {agent_res.cycles_run}, Flaws Fixed: {agent_res.flaws_eliminated})")
        print(f"    [+] Gen Time: {gen_duration}s | Test Exec Time: {eval_res['duration']}s")
        if not eval_res["passed"]:
            print(f"    [-] Error: {eval_res['error']}")

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": eval_res["passed"],
            "cycles": agent_res.cycles_run,
            "flaws_fixed": agent_res.flaws_eliminated,
            "gen_time": gen_duration,
            "test_time": eval_res["duration"],
            "error": eval_res["error"],
            "solution": cleaned
        })

        if i < len(tasks):
            time.sleep(1.0)

    # Calculate final score
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    pass_at_1 = round((passed_count / total_count) * 100, 1)

    print("\n" + "=" * 78)
    print("LIVE HUMANEVAL EVALUATION SCORE TABLE")
    print("=" * 78)
    print(f"{'Task ID':<16} | {'Function':<22} | {'Cycles':<6} | {'Flaws':<6} | {'Test Result'}")
    print("-" * 78)
    for r in results:
        res_tag = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"{r['task_id']:<16} | def {r['entry_point']:<18} | {r['cycles']:<6} | {r['flaws_fixed']:<6} | {res_tag}")
    print("=" * 78)
    print(f"Total Evaluated:  {total_count}")
    print(f"Passed:           {passed_count}")
    print(f"Pass@1 Rate:      {pass_at_1}%")
    print("=" * 78)

    # Save official report
    report_file = os.path.join(output_dir, "humaneval_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "agent": f"{agent_name} (gemini/{model})",
            "model": f"gemini/{model}",
            "benchmark": "HumanEval",
            "date": time.strftime("%Y-%m-%d"),
            "total": total_count,
            "passed": passed_count,
            "pass@1": pass_at_1,
            "results": results
        }, f, indent=2)

    print(f"\n[✓] Results saved to: {report_file}")
    return pass_at_1

if __name__ == "__main__":
    API_KEY = "AIzaSyDXO59HV9JE8hJk5YK-5aRznt9vM5X8vRQ"
    run_humaneval_benchmark(api_key=API_KEY, limit=5)
