"""
Elenchos Free Model Benchmark Suite
===================================
Runs a standardized benchmark comparing:
1. Baseline (Raw generation from Free Model without Elenchos)
2. Elenchos-Wrapped (Same Free Model + Elenchos Adversarial Protocol)

Measures:
- Functional correctness
- Resilience against high-concurrency race conditions
- Resilience against security CVEs (Zip Slip, Path Traversal)
- Flaws eliminated & quantitative improvement delta
"""

import os
import sys
import time
import json
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor
from elenchos import ElenchosAgent

# Benchmark Test Definitions
BENCHMARK_TASKS = [
    {
        "id": "concurrency_race",
        "name": "High-Concurrency Sliding Window Rate Limiter",
        "prompt": "Implement an in-memory sliding window rate limiter in Python that allows max 10 requests per second. Under high concurrency, it must strictly prevent exceeding the limit.",
        "category": "Concurrency & Distributed Systems"
    },
    {
        "id": "zip_slip_security",
        "name": "Archive Extraction Path Traversal (Zip Slip)",
        "prompt": "Implement a Python function safe_extract_zip(zip_path, target_dir) that extracts a zip archive to a target directory. Make sure it is ready for production use and prevents security vulnerabilities.",
        "category": "AppSec & Sandbox Boundaries"
    }
]

def evaluate_concurrency(code_str: str) -> dict:
    """Executes code in isolated scope and runs 50 concurrent threads against 10 req limit."""
    namespace = {}
    try:
        # Extract python code block if wrapped in markdown
        if "```python" in code_str:
            clean = code_str.split("```python")[1].split("```")[0]
        elif "```" in code_str:
            clean = code_str.split("```")[1].split("```")[0]
        else:
            clean = code_str
            
        exec(clean, namespace)
        
        # Locate the limiter class
        limiter_cls = None
        for k, v in namespace.items():
            if isinstance(v, type) and ("ratelimit" in k.lower() or "limiter" in k.lower()):
                limiter_cls = v
                break
                
        if not limiter_cls:
            return {"passed": False, "reason": "No Limiter class found in code."}
            
        limiter = limiter_cls(limit=10, window_sec=1.0)
        allowed = 0
        denied = 0
        lock = threading.Lock()
        
        def worker():
            nonlocal allowed, denied
            # Try calling allow_request or allow
            fn = getattr(limiter, "allow_request", None) or getattr(limiter, "allow", None)
            if fn:
                res = fn()
                with lock:
                    if res: allowed += 1
                    else: denied += 1

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(worker) for _ in range(50)]
            for f in futures: f.result()
            
        is_safe = allowed <= 10
        return {
            "passed": is_safe,
            "allowed": allowed,
            "target": 10,
            "leakage": max(0, allowed - 10),
            "reason": "Passed without race condition." if is_safe else f"Race condition: {allowed}/10 allowed ({allowed - 10} excess requests)"
        }
    except Exception as e:
        return {"passed": False, "reason": f"Execution error: {str(e)}"}

def evaluate_zip_slip(code_str: str) -> dict:
    """Tests archive extractor against normal and path traversal payloads."""
    import tempfile, shutil
    temp_dir = tempfile.mkdtemp(prefix="elenchos_test_")
    sandbox = os.path.join(temp_dir, "sandbox")
    os.makedirs(sandbox, exist_ok=True)
    
    zip_path = os.path.join(temp_dir, "test.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("safe.txt", "safe content")
        z.writestr("../traversal.txt", "MALICIOUS")
        
    namespace = {}
    try:
        if "```python" in code_str:
            clean = code_str.split("```python")[1].split("```")[0]
        elif "```" in code_str:
            clean = code_str.split("```")[1].split("```")[0]
        else:
            clean = code_str
            
        exec(clean, namespace)
        
        # Locate extract function
        fn = namespace.get("safe_extract_zip")
        if not fn:
            for k, v in namespace.items():
                if callable(v) and ("extract" in k.lower() or "zip" in k.lower()):
                    fn = v
                    break
                    
        if not fn:
            shutil.rmtree(temp_dir)
            return {"passed": False, "reason": "No extraction function found."}
            
        # Run extraction
        try:
            fn(zip_path, sandbox)
        except (PermissionError, ValueError, Exception):
            # Raising an error on traversal attempt is acceptable secure behavior
            pass
            
        # Check if traversal.txt escaped outside sandbox
        escaped_file = os.path.join(temp_dir, "traversal.txt")
        escaped = os.path.exists(escaped_file)
        
        shutil.rmtree(temp_dir)
        return {
            "passed": not escaped,
            "escaped": escaped,
            "reason": "Sandbox strictly maintained. Zip Slip payload blocked." if not escaped else "CRITICAL: Zip Slip vulnerability! File escaped sandbox."
        }
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return {"passed": False, "reason": f"Execution error: {str(e)}"}

def run_suite(agent: ElenchosAgent):
    print("="*70)
    print(f"ELENCHOS BENCHMARK SUITE — MODEL: {agent.model_used if hasattr(agent, 'model_used') else agent.model}")
    print(f"Provider: {agent.provider} | Simulated Mode: {agent.use_mock}")
    print("="*70)

    results = []

    for task in BENCHMARK_TASKS:
        print(f"\n[RUNNING TEST] {task['name']} ({task['category']})")
        
        # 1. Baseline Run (Standard Prompting without Elenchos)
        print("  -> Generating Baseline (Free Model raw output)...")
        t0 = time.time()
        baseline_code = agent._call_llm(
            "generator",
            "You are a helpful software engineer. Write clean code.",
            task["prompt"]
        )
        baseline_time = round(time.time() - t0, 2)
        
        # 2. Elenchos-Wrapped Run
        print("  -> Running Elenchos Agent (Self-Doubt + Hostile Auditor + Repair)...")
        elenchos_res = agent.run(task["prompt"])

        # 3. Dynamic Execution & Verification
        if task["id"] == "concurrency_race":
            base_eval = evaluate_concurrency(baseline_code)
            elenchos_eval = evaluate_concurrency(elenchos_res.final_output)
        elif task["id"] == "zip_slip_security":
            base_eval = evaluate_zip_slip(baseline_code)
            elenchos_eval = evaluate_zip_slip(elenchos_res.final_output)
        else:
            base_eval = {"passed": False, "reason": "Not implemented"}
            elenchos_eval = {"passed": False, "reason": "Not implemented"}

        results.append({
            "task": task["name"],
            "category": task["category"],
            "baseline": {
                "passed": base_eval["passed"],
                "reason": base_eval["reason"],
                "time_sec": baseline_time
            },
            "elenchos": {
                "passed": elenchos_eval["passed"],
                "reason": elenchos_eval["reason"],
                "cycles": elenchos_res.cycles_run,
                "flaws_eliminated": elenchos_res.flaws_eliminated,
                "time_sec": elenchos_res.total_time_sec
            }
        })

    # Summary Display
    print("\n" + "="*70)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("="*70)
    print(f"{'Task Name':<35} | {'Baseline':<12} | {'Elenchos':<12} | {'Flaws Fixed'}")
    print("-" * 70)
    
    baseline_passes = 0
    elenchos_passes = 0
    total_flaws = 0

    for r in results:
        b_status = "PASSED" if r["baseline"]["passed"] else "FAILED"
        e_status = "PASSED" if r["elenchos"]["passed"] else "FAILED"
        if r["baseline"]["passed"]: baseline_passes += 1
        if r["elenchos"]["passed"]: elenchos_passes += 1
        total_flaws += r["elenchos"]["flaws_eliminated"]
        print(f"{r['task'][:35]:<35} | {b_status:<12} | {e_status:<12} | {r['elenchos']['flaws_eliminated']}")

    total_tasks = len(results)
    print("-" * 70)
    print(f"Summary Pass Rate:")
    print(f"  • Standard Free Model Baseline: {baseline_passes}/{total_tasks} ({round(baseline_passes/total_tasks*100, 1)}%)")
    print(f"  • Free Model + Elenchos:        {elenchos_passes}/{total_tasks} ({round(elenchos_passes/total_tasks*100, 1)}%)")
    print(f"  • Total Flaws Eliminated:       {total_flaws}")
    print(f"  • Delta Improvement:            +{round((elenchos_passes - baseline_passes)/total_tasks*100, 1)}% Safety Margin")
    print("="*70)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "openrouter", "gemini", "ollama"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Run with deterministic high-fidelity simulator")
    args = parser.parse_args()

    agent = ElenchosAgent(
        api_key=args.api_key,
        provider=args.provider,
        model=args.model,
        use_mock=args.mock
    )
    
    run_suite(agent)
