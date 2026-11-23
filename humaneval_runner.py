"""
Alenchos HumanEval Benchmark Harness (Full Test Edition)
=========================================================
Runs Alenchos (gemini-3.1-flash-lite) across official HumanEval tasks,
executing live Python test suites with immediate verification.
"""

import os
import sys
import json
import time
import re
import requests
import argparse
from typing import List, Dict, Any, Optional
from elenchos import ElenchosAgent

HUMANEVAL_HF_API = "https://datasets-server.huggingface.co/rows"

def fetch_all_humaneval_tasks(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Pulls task instances from openai/openai_humaneval on Hugging Face."""
    print(f"[*] Fetching up to {limit} tasks from openai/openai_humaneval (split: test, offset: {offset})...", flush=True)
    all_tasks = []
    current_offset = offset
    batch_size = 100

    while len(all_tasks) < limit:
        to_fetch = min(batch_size, limit - len(all_tasks))
        params = {
            "dataset": "openai/openai_humaneval",
            "config": "openai_humaneval",
            "split": "test",
            "offset": current_offset,
            "limit": to_fetch
        }
        try:
            resp = requests.get(HUMANEVAL_HF_API, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                rows = [row["row"] for row in data.get("rows", [])]
                if not rows:
                    break
                all_tasks.extend(rows)
                current_offset += len(rows)
                if len(rows) < to_fetch:
                    break
            else:
                print(f"[!] HF API returned status {resp.status_code}.", flush=True)
                break
        except Exception as e:
            print(f"[!] Warning: HF API error ({e}).", flush=True)
            break

    print(f"[+] Successfully fetched {len(all_tasks)} verified HumanEval tasks.", flush=True)
    return all_tasks[:limit]

def clean_code(raw_code: str, entry_point: str) -> str:
    """Extracts pure executable Python code from markdown blocks if present."""
    if "```python" in raw_code:
        parts = raw_code.split("```python")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    elif "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return raw_code.strip()

def test_solution(prompt: str, solution: str, test_code: str, entry_point: str) -> Dict[str, Any]:
    """Executes the test suite against the generated solution in an isolated environment."""
    exec_scope = {}
    full_program = f"{prompt}\n{solution}\n{test_code}\ncheck({entry_point})\n"
    
    t0 = time.time()
    try:
        exec(full_program, exec_scope)
        duration = round(time.time() - t0, 3)
        return {"passed": True, "error": None, "duration": duration}
    except AssertionError:
        duration = round(time.time() - t0, 3)
        return {"passed": False, "error": "AssertionError: Test assertion failed", "duration": duration}
    except Exception as e:
        duration = round(time.time() - t0, 3)
        return {"passed": False, "error": f"{type(e).__name__}: {str(e)[:100]}", "duration": duration}

def run_humaneval_benchmark(
    api_key: str,
    model: str = "gemini-3.1-flash-lite",
    limit: int = 25,
    offset: int = 0,
    agent_name: str = "Alenchos"
):
    print("=" * 80, flush=True)
    print("ALENCHOS HUMANEVAL OFFICIAL BENCHMARK RUNNER", flush=True)
    print("=" * 80, flush=True)
    print(f"Agent Framework:     {agent_name} Socratic Quality Engine", flush=True)
    print(f"Backend Model:       gemini/{model}", flush=True)
    print(f"Benchmark:           HumanEval (OpenAI / EvalPlus)", flush=True)
    print(f"Target Tasks:        {limit} problems (offset: {offset})", flush=True)
    print("=" * 80, flush=True)

    agent = ElenchosAgent(
        api_key=api_key,
        provider="gemini",
        model=model,
        max_cycles=2,
        use_mock=False
    )

    tasks = fetch_all_humaneval_tasks(limit=limit, offset=offset)
    if not tasks:
        print("[!] No tasks retrieved from Hugging Face.", flush=True)
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "humaneval_output")
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "humaneval_results.json")

    results = []

    # Load existing if available
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                results = existing.get("results", [])
        except Exception:
            results = []

    already_done_ids = {r["task_id"] for r in results}

    print(f"\n[*] Starting evaluation (already completed: {len(already_done_ids)} tasks)...\n", flush=True)

    for i, t in enumerate(tasks, 1):
        task_id = t["task_id"]
        prompt = t["prompt"]
        entry_point = t["entry_point"]
        test_code = t["test"]

        if task_id in already_done_ids:
            print(f"[{i}/{len(tasks)}] {task_id} already evaluated — skipping.", flush=True)
            continue

        print(f"[{i}/{len(tasks)}] {task_id} -> def {entry_point}(...)", flush=True)

        socratic_prompt = (
            f"You are a rigorous Python software engineer.\n"
            f"Complete the following Python function correctly, handling all boundary, type, and edge cases.\n"
            f"Return ONLY the implementation code for {entry_point} with matching types.\n\n"
            f"{prompt}"
        )

        t_start = time.time()
        agent_res = agent.run(socratic_prompt)
        gen_duration = round(time.time() - t_start, 2)

        cleaned = clean_code(agent_res.final_output, entry_point)

        # Run live unit test execution
        eval_res = test_solution(prompt, cleaned, test_code, entry_point)

        status_str = "PASSED" if eval_res["passed"] else "FAILED"
        print(f"    [+] Status: {status_str} (Cycles: {agent_res.cycles_run}, Flaws Fixed: {agent_res.flaws_eliminated})", flush=True)
        print(f"    [+] Gen Time: {gen_duration}s | Test Exec Time: {eval_res['duration']}s", flush=True)
        if not eval_res["passed"]:
            print(f"    [-] Error: {eval_res['error']}", flush=True)

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

        # Save progress after every task
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)
        pass_at_1 = round((passed_count / total_count) * 100, 1)

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

        time.sleep(1.0)

    # Final summary table
    print("\n" + "=" * 80, flush=True)
    print("LIVE HUMANEVAL FINAL SCORE TABLE", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Task ID':<16} | {'Function':<22} | {'Cycles':<6} | {'Flaws':<6} | {'Test Result'}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        res_tag = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"{r['task_id']:<16} | def {r['entry_point']:<18} | {r['cycles']:<6} | {r['flaws_fixed']:<6} | {res_tag}", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Evaluated:  {total_count}", flush=True)
    print(f"Passed:           {passed_count}", flush=True)
    print(f"Pass@1 Rate:      {pass_at_1}%", flush=True)
    print("=" * 80, flush=True)

    return pass_at_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HumanEval benchmark")
    parser.add_argument("--limit", type=int, default=20, help="Number of tasks to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Starting task index")
    args = parser.parse_args()

    API_KEY = "os.getenv("GEMINI_API_KEY")"
    run_humaneval_benchmark(api_key=API_KEY, limit=args.limit, offset=args.offset)
