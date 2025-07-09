#!/bin/bash

function run_backend_tests() {
    echo "🧪 Starte Testdurchlauf..."
    pytest -m "not heavy" -q || {
        echo "❌ Tests fehlgeschlagen" >&2
        return 1
    }
}
