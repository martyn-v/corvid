"""Run the agent graph over every generated email in the harness."""

import asyncio
import json

from corvid.agent.graph import build_graph
from corvid.contracts import Provenance, QuoteRequest
from corvid.llm import create_model
from harness.models import Case
from harness.paths import CASES_PATH, EMAILS_DIR


def _summarize_provenance(prov: dict[str, Provenance]):
    expected_fields = QuoteRequest.dot_fields()
    provenance_fields = prov.keys()
    missing_fields = set(expected_fields) - set(provenance_fields)
    fields_by_source = {}
    for path, p in prov.items():
        fields_by_source.setdefault(p.source, []).append(path)
    print(f"Provenance: {len(prov)} fields, missing: {len(missing_fields)}")
    for source, paths in fields_by_source.items():
        print(f"  Source {source}: {len(paths)} fields")


async def main():
    model = create_model(format="json", temperature=0)
    graph = build_graph(model)

    with open(CASES_PATH) as f:
        cases = [Case.model_validate(json.loads(line)) for line in f]

    for case in cases:
        result = await graph.ainvoke({"file_path": str(EMAILS_DIR / f"{case.key}.eml")})
        qr = result["quote_request"]
        print(f"{case.key}: {qr.origin} -> {qr.destination}")
        _summarize_provenance(result["provenance"])
        break


if __name__ == "__main__":
    asyncio.run(main())
