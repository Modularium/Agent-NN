#!/bin/bash
echo "🧪 Starte Testdurchlauf..."
pytest -m "not heavy" -q || echo "❌ Tests fehlgeschlagen"
