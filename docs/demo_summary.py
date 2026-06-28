"""Print the headline numbers from a benchmark run JSON file (used by docs/demo.tape)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(strategy: str = "naive_vector") -> None:
    path = Path(f"results/longmemeval-s_{strategy}.json")
    if not path.exists():
        print(f"missing: {path}")
        sys.exit(1)
    data = json.loads(path.read_text())
    f1 = data.get("abstention_f1")
    f1_str = f"{f1:.2f}" if f1 is not None else "—"
    upd = data.get("update_precision")
    upd_str = f"{upd:.2%}" if upd is not None else "—"
    tem = data.get("temporal_correctness")
    tem_str = f"{tem:.2%}" if tem is not None else "—"
    print(f"strategy:   {data['strategy']}")
    print(f"run_id:     {data['run_id']}")
    print(f"questions:  {len(data['recall_records'])}")
    print(f"accuracy:   {data['accuracy']:.2%}")
    print(f"recall@k:   {data['mean_session_recall_at_k']:.2%}")
    print(f"abst F1:    {f1_str} (n={data.get('abstention_n', 0)})")
    print(f"update prec:{upd_str} (n={data.get('update_n', 0)})")
    print(f"temp corr:  {tem_str} (n={data.get('temporal_n', 0)})")
    print(f"cost USD:   ${data['total_cost_usd']:.4f}")
    print(f"avg latency:{data['avg_recall_latency_ms']:.0f}ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "naive_vector")
