"""Run all diagnostic scripts and produce a combined summary."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DIAGNOSTICS_DIR, EVALUATION_DIR
from src.diagnostics_io import save_json_report

SUMMARY_PATH = DIAGNOSTICS_DIR / "diagnostics_run_summary.json"


def _run_step(name: str, fn) -> dict:
    try:
        result = fn()
        return {"step": name, "status": "ok", "detail": result}
    except Exception as exc:
        return {
            "step": name,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc()[-800:],
        }


def main() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    run_log: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    import inspect_dataset as ds_mod

    step1 = _run_step("inspect_dataset", lambda: ds_mod.run_inspection())
    run_log["steps"].append(step1)
    run_log["dataset_status"] = (
        step1["detail"][1] if step1["status"] == "ok" else {"error": step1.get("error")}
    )

    import inspect_ocr as ocr_mod

    step_ocr = _run_step("inspect_ocr", lambda: ocr_mod.run_inspection(full=False))
    run_log["steps"].append(step_ocr)
    run_log["ocr_status"] = (
        step_ocr.get("detail") if step_ocr["status"] == "ok" else {"error": step_ocr.get("error")}
    )

    def build_kb():
        from src.pipeline import build_knowledge_base

        return build_knowledge_base()

    step2 = _run_step("build_knowledge_base", build_kb)
    run_log["steps"].append(step2)
    run_log["build_status"] = (
        step2.get("detail") if step2["status"] == "ok" else {"error": step2.get("error")}
    )

    import inspect_chunks as ch_mod

    step3 = _run_step("inspect_chunks", ch_mod.run_inspection)
    run_log["steps"].append(step3)
    run_log["chunks_status"] = (
        step3.get("detail") if step3["status"] == "ok" else {"error": step3.get("error")}
    )

    import inspect_retrieval as ret_mod

    step4 = _run_step(
        "inspect_retrieval",
        lambda: ret_mod.run_queries(ret_mod.DEFAULT_QUERIES, top_k=5),
    )
    run_log["steps"].append(step4)
    run_log["retrieval_status"] = (
        step4.get("detail", {}).get("overall")
        if step4["status"] == "ok"
        else {"error": step4.get("error")}
    )

    import inspect_evaluation as ev_mod

    step5 = _run_step(
        "inspect_evaluation",
        lambda: ev_mod.run_inspection(
            eval_path=EVALUATION_DIR / "eval_questions.csv"
        ),
    )
    run_log["steps"].append(step5)

    proposed = EVALUATION_DIR / "eval_questions_proposed.csv"
    if proposed.exists():
        step5b = _run_step(
            "inspect_evaluation_proposed",
            lambda: ev_mod.run_inspection(eval_path=proposed),
        )
        run_log["steps"].append(step5b)

    proposed_ocr = EVALUATION_DIR / "eval_questions_proposed_with_ocr.csv"
    if proposed_ocr.exists():
        step5c = _run_step(
            "inspect_evaluation_proposed_ocr",
            lambda: ev_mod.run_inspection(eval_path=proposed_ocr),
        )
        run_log["steps"].append(step5c)

    run_log["evaluation_status"] = (
        step5.get("detail") if step5["status"] == "ok" else {"error": step5.get("error")}
    )

    import inspect_llm as llm_mod

    step6 = _run_step("inspect_llm", llm_mod.run_inspection)
    run_log["steps"].append(step6)
    run_log["llm_status"] = (
        step6.get("detail") if step6["status"] == "ok" else {"error": step6.get("error")}
    )

    if DIAGNOSTICS_DIR.exists():
        run_log["files_generated"] = sorted(
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in DIAGNOSTICS_DIR.rglob("*")
            if p.is_file()
        )

    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_json_report(run_log, SUMMARY_PATH)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS RUN SUMMARY")
    print("=" * 60)
    ds = run_log.get("dataset_status", {})
    print(f"Dataset: {ds.get('conclusion', ds)}")
    ocr = run_log.get("ocr_status", {})
    if isinstance(ocr, dict):
        deps = ocr.get("ocr_dependencies", {})
        print(f"OCR ready: {deps.get('ready')} — {deps.get('message', '')[:120]}")
    ch = run_log.get("chunks_status", {})
    print(f"Chunks: {ch.get('recommendation', ch) if isinstance(ch, dict) else ch}")
    print(f"Retrieval: {run_log.get('retrieval_status')}")
    print(f"Evaluation: {run_log.get('evaluation_status')}")
    llm = run_log.get("llm_status", {})
    print(f"LLM/Gemini: {llm.get('status') if isinstance(llm, dict) else llm}")
    if llm.get("warnings"):
        for w in llm["warnings"]:
            print(f"  WARNING: {w}")
    print(f"\nFiles generated: {len(run_log.get('files_generated', []))}")
    print(f"Full log: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
