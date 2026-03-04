from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(url="http://localhost:6333")
sigs = set()
offset = None
while True:
    pts, next_off = client.scroll(
        collection_name="uodo_decisions",
        scroll_filter=Filter(must=[
            FieldCondition(key="doc_type", match=MatchValue(value="uodo_decision"))
        ]),
        limit=500, offset=offset,
        with_payload=["signature"], with_vectors=False,
    )
    for p in pts:
        sigs.add(p.payload.get("signature", ""))
    if not next_off:
        break
    offset = next_off

print(f"Unikalne sygnatury: {len(sigs)}")
