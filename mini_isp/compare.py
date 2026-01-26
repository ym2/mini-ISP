from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CompareBundle:
    schema_version: str
    title: str
    created_utc: str
    a_label: str
    a_run_dir: str
    b_label: str
    b_run_dir: str
    notes: str
    stage_match_primary: str
    stage_match_fallback: str


def parse_compare_bundle(bundle: Dict[str, Any]) -> CompareBundle:
    stage_match = bundle.get("stage_match", {})
    return CompareBundle(
        schema_version=bundle["schema_version"],
        title=bundle.get("title", ""),
        created_utc=bundle.get("created_utc", ""),
        a_label=bundle["a"]["label"],
        a_run_dir=bundle["a"]["run_dir"],
        b_label=bundle["b"]["label"],
        b_run_dir=bundle["b"]["run_dir"],
        notes=bundle.get("notes", ""),
        stage_match_primary=stage_match.get("primary", "index"),
        stage_match_fallback=stage_match.get("fallback", "name"),
    )


def match_stages(
    stages_a: List[Dict[str, Any]], stages_b: List[Dict[str, Any]]
) -> List[Dict[str, Optional[Dict[str, Any]]]]:
    matches: List[Dict[str, Optional[Dict[str, Any]]]] = []
    used_b: set[int] = set()

    for stage_a in stages_a:
        stage_b = None
        idx = stage_a.get("index")
        if isinstance(idx, int):
            for i, cand in enumerate(stages_b):
                if i == idx and cand.get("index") == idx and cand.get("name") == stage_a.get("name"):
                    stage_b = cand
                    used_b.add(i)
                    break
        if stage_b is None:
            for i, cand in enumerate(stages_b):
                if i in used_b:
                    continue
                if cand.get("name") == stage_a.get("name"):
                    stage_b = cand
                    used_b.add(i)
                    break
        matches.append({"a": stage_a, "b": stage_b})

    for i, stage_b in enumerate(stages_b):
        if i not in used_b:
            matches.append({"a": None, "b": stage_b})

    return matches


def build_compare_bundle(
    run_dir_a: str,
    run_dir_b: str,
    label_a: str,
    label_b: str,
    title: str,
    created_utc: str,
    notes: str,
) -> Dict[str, Any]:
    return {
        "schema_version": "0.1",
        "title": title,
        "created_utc": created_utc,
        "a": {"label": label_a, "run_dir": run_dir_a},
        "b": {"label": label_b, "run_dir": run_dir_b},
        "notes": notes,
        "stage_match": {"primary": "index", "fallback": "name"},
    }


def _require_manifest(run_dir: str) -> None:
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.json not found in {run_dir}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="mini-ISP compare bundle helper")
    parser.add_argument("--a", dest="run_dir_a", help="Run directory A")
    parser.add_argument("--b", dest="run_dir_b", help="Run directory B")
    parser.add_argument("--out", dest="out_path", help="Output compare bundle path")
    parser.add_argument("--label-a", dest="label_a", help="Label for A")
    parser.add_argument("--label-b", dest="label_b", help="Label for B")
    parser.add_argument("--notes", dest="notes", default="", help="Notes (optional)")
    parser.add_argument("--title", dest="title", default="mini-ISP compare", help="Title (optional)")
    parser.add_argument(
        "--created-utc", dest="created_utc", default=None, help="Created UTC ISO8601"
    )
    parser.add_argument("--init", dest="init_path", default=None, help="Init template copy path")
    args = parser.parse_args()

    if args.init_path:
        init_path = Path(args.init_path)
        example = Path(__file__).resolve().parent.parent / "examples" / "compare_bundle.example.json"
        if init_path.exists():
            raise FileExistsError(f"{init_path} already exists")
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Created {init_path}. Open viewer with ?compare=/{init_path.as_posix()}")
        return

    if not all([args.run_dir_a, args.run_dir_b, args.out_path, args.label_a, args.label_b]):
        raise SystemExit("Missing required args: --a --b --out --label-a --label-b")

    _require_manifest(args.run_dir_a)
    _require_manifest(args.run_dir_b)

    created = args.created_utc or _utc_now_iso()
    bundle = build_compare_bundle(
        run_dir_a=args.run_dir_a,
        run_dir_b=args.run_dir_b,
        label_a=args.label_a,
        label_b=args.label_b,
        title=args.title,
        created_utc=created,
        notes=args.notes or "",
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
