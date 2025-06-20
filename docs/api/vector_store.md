# Vector Store API

`POST /document` – Add a document with optional `id` and `text` fields. Returns the stored document id.

`POST /query` – Submit a text query and receive a list of similar documents sorted by score.

`GET /health` – Simple health check.


### Beispiel

Request:
```bash
curl -X POST http://localhost:8003/query \
     -H "Content-Type: application/json" \
     -d '{"text": "hello"}'
```

Response:
```json
[
    {"id": "doc1", "score": 0.9}
]
```
