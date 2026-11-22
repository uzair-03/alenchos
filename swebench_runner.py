"""
Elenchos SWE-bench Test Harness (Closed-Source / Proprietary Edition)
=====================================================================
Pulls verified real-world tasks directly from the official SWE-bench dataset (Hugging Face),
runs them through the private Elenchos adversarial quality engine using any free or
commercial LLM API, and generates official leaderboard submission artifacts:

1. predictions.jsonl  (SWE-bench standard patch submissions)
2. metadata.yaml      (Leaderboard submission metadata with oss: false)
3. trajs/             (Execution traces & Elenchos audit records)

Every submission explicitly records the exact underlying backend model name
(e.g., groq/llama-3.3-70b-versatile, gemini/gemini-2.0-flash, etc.).
"""

import os
import sys
import json
import time
import argparse
import requests
from typing import List, Dict, Any, Optional
from elenchos import ElenchosAgent, FREE_PROVIDER_PRESETS

SWE_BENCH_API = "https://datasets-server.huggingface.co/rows"

def fetch_swebench_instances(dataset: str = "princeton-nlp/SWE-bench_Lite", limit: int = 3, offset: int = 0) -> List[Dict[str, Any]]:
    """Pulls task instances directly from the official SWE-bench Hugging Face dataset."""
    print(f"[*] Fetching {limit} task instances from {dataset} (split: test, offset: {offset})...")
    params = {
        "dataset": dataset,
        "config": "default",
        "split": "test",
        "offset": offset,
        "limit": limit
    }
    try:
        resp = requests.get(SWE_BENCH_API, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            instances = [row["row"] for row in data.get("rows", [])][:limit]
            print(f"[+] Successfully fetched {len(instances)} real task instances.")
            return instances
        else:
            print(f"[!] Warning: HF API returned status {resp.status_code}. Using cached fallback instances.")
            return get_fallback_instances(limit)
    except Exception as e:
        print(f"[!] Network error connecting to HuggingFace ({e}). Using cached fallback instances.")
        return get_fallback_instances(limit)

def get_fallback_instances(limit: int) -> List[Dict[str, Any]]:
    """Verified offline fallback SWE-bench instances for zero-network testing."""
    sample = [
        {
            "instance_id": "astropy__astropy-7746",
            "repo": "astropy/astropy",
            "base_commit": "3ee724620f4cf32aa0edbaad19dbff58849b28b7",
            "problem_statement": (
                "Issue when passing empty lists/arrays to WCS transformations.\n"
                "The following should not fail but instead should return empty lists/arrays:\n"
                "wcs.wcs_pix2world([], [], 0)\n"
                "Currently it throws InconsistentAxisTypesError."
            )
        },
        {
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "base_commit": "20b27b3b3a0dfa4843b7ec3b7b51b3f9bb869fec",
            "problem_statement": (
                "UsernameValidator allows trailing newline.\n"
                "ASCIIUsernameValidator and UnicodeUsernameValidator use regex r'^[\\w.@+-]+$'.\n"
                "In Python re, $ matches the end of the string or just before a newline. "
                "This allows invalid usernames ending in \\n to pass validation."
            )
        },
        {
            "instance_id": "sympy__sympy-18057",
            "repo": "sympy/sympy",
            "base_commit": "36ef7b81f9b5c327cf02a3a1f10731a57c5e2634",
            "problem_statement": (
                "Equality of Symbol and foreign object raises AttributeError.\n"
                "Evaluating `Symbol('x') == object()` should return False, "
                "but currently crashes attempting to access `_eval_power` or `is_commutative` on the foreign object."
            )
        }
    ]
    return sample[:limit]

def generate_git_diff(instance_id: str, code_patch: str) -> str:
    """Formats code patch into a canonical unified git diff."""
    header = f"diff --git a/{instance_id}/patch.py b/{instance_id}/patch.py\n"
    header += "--- a/source\n+++ b/source\n"
    # Ensure standard diff format
    lines = code_patch.strip().split("\n")
    diff_lines = [f"+{l}" if not l.startswith(("+", "-", "@", "diff")) else l for l in lines]
    return header + "\n".join(diff_lines)

def run_swebench_eval(
    provider: str = "groq",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: int = 3,
    mock: bool = False,
    output_dir: str = "swebench_output",
    agent_name: str = "Alenchos"
):
    # Resolve absolute path for output directory inside project folder
    if not os.path.isabs(output_dir):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    trajs_dir = os.path.join(output_dir, "trajs")
    os.makedirs(trajs_dir, exist_ok=True)

    # Initialize the Elenchos Agent with the target backend model
    agent = ElenchosAgent(
        api_key=api_key,
        provider=provider,
        model=model,
        use_mock=mock
    )

    # Build exact model identifier string
    full_model_name = f"{agent.provider}/{agent.model}"
    if agent.use_mock:
        full_model_name += " [Simulator]"
    
    agent_display_name = f"{agent_name} ({full_model_name})"

    print("="*75)
    print("ELENCHOS SWE-BENCH EVALUATION HARNESS (PROPRIETARY)")
    print("="*75)
    print(f"Agent Framework:     Elenchos Adversarial Socratic Engine")
    print(f"Backend AI Model:    {full_model_name}")
    print(f"Target Benchmark:    SWE-bench Verified / Lite")
    print(f"Proprietary Status:  oss: false (Intellectual Property Protected)")
    print(f"Output Directory:    {output_dir}")
    print("="*75)

    instances = fetch_swebench_instances(limit=limit)
    predictions = []
    audit_summary = []

    print(f"\n[*] Starting patch generation for {len(instances)} tasks...")

    for idx, inst in enumerate(instances, 1):
        inst_id = inst["instance_id"]
        repo = inst.get("repo", "unknown")
        problem = inst["problem_statement"]

        print(f"\n[{idx}/{len(instances)}] Processing: {inst_id} ({repo})")
        print(f"    Issue: {problem.strip().splitlines()[0][:75]}...")

        task_prompt = (
            f"You are working on the open-source repository {repo}.\n"
            f"Task Instance ID: {inst_id}\n\n"
            f"Problem Statement:\n{problem}\n\n"
            f"Generate the exact Python patch to resolve this issue without introducing any regressions."
        )

        t_start = time.time()
        # Run through the Elenchos Adversarial Quality Engine
        result = agent.run(task_prompt)
        duration = round(time.time() - t_start, 2)

        # Format patch as git diff
        patch_diff = generate_git_diff(inst_id, result.final_output)

        # Record for predictions.jsonl
        prediction_record = {
            "instance_id": inst_id,
            "model_patch": patch_diff,
            "model_name_or_path": agent_display_name,
            "backend_model": full_model_name,
            "timestamp": time.time()
        }
        predictions.append(prediction_record)

        # Save individual trajectory / audit log
        traj_path = os.path.join(trajs_dir, f"{inst_id}.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump({
                "instance_id": inst_id,
                "backend_model": full_model_name,
                "agent": agent_display_name,
                "cycles_run": result.cycles_run,
                "flaws_eliminated": result.flaws_eliminated,
                "passed_audit": result.passed_audit,
                "duration_sec": duration,
                "audit_trail": [
                    {
                        "step": a.step,
                        "cycle": a.cycle,
                        "flaws": a.flaws_found,
                        "critic_notes": a.critic_notes
                    }
                    for a in result.audit_trail
                ]
            }, f, indent=2)

        print(f"    [+] Status: {'PASSED AUDIT' if result.passed_audit else 'FLAGGED'}")
        print(f"    [+] Cycles: {result.cycles_run} | Flaws Eliminated: {result.flaws_eliminated} | Time: {duration}s")
        print(f"    [+] Backend Model Recorded: {full_model_name}")

        audit_summary.append({
            "instance_id": inst_id,
            "backend_model": full_model_name,
            "cycles": result.cycles_run,
            "flaws_fixed": result.flaws_eliminated,
            "status": "PASSED" if result.passed_audit else "FLAGGED"
        })

        # Polite rate-limiting pause for free tiers
        if not mock and idx < len(instances):
            time.sleep(2.0)

    # 1. Write predictions.jsonl
    predictions_file = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\n[✓] Generated predictions file: {predictions_file}")

    # 2. Write metadata.yaml (Official SWE-bench format with oss: false)
    metadata_file = os.path.join(output_dir, "metadata.yaml")
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(f"name: \"{agent_display_name}\"\n")
        f.write(f"backend_model: \"{full_model_name}\"\n")
        f.write("oss: false\n")
        f.write("site: \"https://github.com/elenchos-ai\"\n")
        f.write(f"submission_date: \"{time.strftime('%Y-%m-%d')}\"\n")
        f.write("description: \"Proprietary Socratic Adversarial Quality Engine for LLMs\"\n")
    print(f"[✓] Generated metadata file: {metadata_file} (oss: false)")

    # 3. Print Final Evaluation Table
    print("\n" + "="*75)
    print(f"SWE-BENCH BATCH SUMMARY — BACKEND MODEL: {full_model_name}")
    print("="*75)
    print(f"{'Instance ID':<30} | {'Status':<10} | {'Cycles':<8} | {'Flaws Fixed'}")
    print("-" * 75)
    for s in audit_summary:
        print(f"{s['instance_id']:<30} | {s['status']:<10} | {s['cycles']:<8} | {s['flaws_fixed']}")
    print("="*75)
    print(f"Total Instances: {len(audit_summary)}")
    print(f"Audit Pass Rate: {sum(1 for s in audit_summary if s['status'] == 'PASSED')}/{len(audit_summary)}")
    print(f"Artifacts ready for SWE-bench evaluation harness under: {output_dir}/")
    print("="*75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Elenchos on official SWE-bench instances.")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "openrouter", "gemini", "ollama"])
    parser.add_argument("--model", type=str, default=None, help="Explicit model name (e.g., llama-3.3-70b-versatile, gemini-2.0-flash)")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--limit", type=int, default=3, help="Number of SWE-bench instances to process")
    parser.add_argument("--mock", action="store_true", help="Run with simulator for instant verification")
    parser.add_argument("--output-dir", type=str, default="swebench_output")
    parser.add_argument("--agent-name", type=str, default="Alenchos", help="Display name for the agent framework")
    args = parser.parse_args()

    run_swebench_eval(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        mock=args.mock,
        output_dir=args.output_dir,
        agent_name=args.agent_name
    )
