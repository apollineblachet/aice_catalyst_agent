from __future__ import annotations
import argparse
from pathlib import Path
from agent.runner import run_agent
from agent.io_utils import write_text, save_plan_json

def main():
    ap = argparse.ArgumentParser(description="Run the planning agent on a requirement.")
    ap.add_argument("--input", "-i", required=True, help="Path to a .md/.txt file or raw text")
    ap.add_argument("--outdir", "-o", default="examples/outputs", help="Directory for outputs")
    ap.add_argument("--name", "-n", default="plan", help="Output base name (no extension)")
    ap.add_argument("--stream", action="store_true", help="Print streaming events")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = outdir / args.name

    raw_text, plan = run_agent(args.input, stream=args.stream)
    txt_path  = write_text(str(base.with_suffix(".txt")), raw_text)
    json_path = save_plan_json(plan, str(base.with_suffix(".json")))
    print(f"✅ Wrote:\n  {txt_path}\n  {json_path}")

if __name__ == "__main__":
    main()
