#!/bin/bash
# -*- coding: utf-8 -*-
# Umfassendes Test-Framework für Agent-NN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
source "$SCRIPT_DIR/lib/status_utils.sh"
source "$SCRIPT_DIR/helpers/common.sh"

# Test-Konfiguration
TEST_PARALLEL=true
TEST_COVERAGE=false
TEST_VERBOSE=false
TEST_QUICK=false
TEST_INTEGRATION=false
TEST_E2E=false
SAVE_ARTIFACTS=false
CLEANUP_AFTER=true

# Test-Kategorien
declare -A TEST_CATEGORIES=(
    [unit]="Unit Tests (schnell, isoliert)"
    [integration]="Integration Tests (Services zusammen)"
    [e2e]="End-to-End Tests (vollständige Workflows)"
    [performance]="Performance Tests (Benchmarks)"
    [security]="Security Tests (Sicherheitsüberprüfungen)"
    [api]="API Tests (REST/HTTP Endpoints)"
    [frontend]="Frontend Tests (UI/Browser)"
    [mcp]="MCP Tests (Model Context Protocol)"
)

# Test-Ergebnisse
declare -A TEST_RESULTS=(
    [total]=0
    [passed]=0
    [failed]=0
    [skipped]=0
    [errors]=0
)

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [CATEGORIES...]

Umfassendes Test-Framework für Agent-NN

OPTIONS:
    --coverage          Code-Coverage Analyse aktivieren
    --verbose           Ausführliche Test-Ausgabe
    --quick             Nur schnelle Tests ausführen
    --parallel          Tests parallel ausführen (default)
    --no-parallel       Tests sequenziell ausführen
    --integration       Integration Tests einschließen
    --e2e               End-to-End Tests einschließen
    --save-artifacts    Test-Artefakte speichern
    --no-cleanup        Keine Bereinigung nach Tests
    --timeout SECONDS   Timeout für Tests (default: 300)
    --output FORMAT     Ausgabeformat (junit|tap|json|text)
    --filter PATTERN    Nur Tests mit Pattern ausführen
    -h, --help          Diese Hilfe anzeigen

CATEGORIES:
    unit                Unit Tests (Standard)
    integration         Integration Tests
    e2e                 End-to-End Tests
    performance         Performance Tests
    security            Security Tests
    api                 API Tests
    frontend            Frontend Tests
    mcp                 MCP Tests
    
    all                 Alle verfügbaren Tests
    fast                Nur schnelle Tests (unit)
    full                Komplette Test-Suite

BEISPIELE:
    $(basename "$0")                        # Standard Unit Tests
    $(basename "$0") --coverage --verbose   # Mit Coverage und Details
    $(basename "$0") unit integration       # Unit und Integration Tests
    $(basename "$0") --e2e --save-artifacts # E2E Tests mit Artefakt-Speicherung
    $(basename "$0") --quick --filter "test_api*" # Schnelle API Tests

EOF
}

# Test-Setup
setup_test_environment() {
    log_info "Bereite Test-Umgebung vor..."
    
    # Test-Verzeichnisse erstellen
    local test_dirs=(
        "$REPO_ROOT/test-results"
        "$REPO_ROOT/test-artifacts"
        "$REPO_ROOT/test-coverage"
    )
    
    for dir in "${test_dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    # Test-Datenbank vorbereiten
    if [[ "$TEST_INTEGRATION" == "true" || "$TEST_E2E" == "true" ]]; then
        setup_test_database
    fi
    
    # Test-Services starten falls nötig
    if [[ "$TEST_INTEGRATION" == "true" || "$TEST_E2E" == "true" ]]; then
        start_test_services
    fi
    
    log_ok "Test-Umgebung bereit"
}

# Test-Datenbank Setup
setup_test_database() {
    log_debug "Bereite Test-Datenbank vor..."
    
    # Test-DB Konfiguration
    export DATABASE_URL="postgresql://test_user:test_pass@localhost:5433/test_db"
    export REDIS_URL="redis://localhost:6380/1"
    
    # Test-Container starten falls nicht laufend
    if ! docker ps --format "{{.Names}}" | grep -q "test-postgres"; then
        docker run -d --name test-postgres \
            -e POSTGRES_DB=test_db \
            -e POSTGRES_USER=test_user \
            -e POSTGRES_PASSWORD=test_pass \
            -p 5433:5432 \
            postgres:14-alpine >/dev/null 2>&1
        
        # Warte bis DB bereit
        timeout 30 bash -c 'until docker exec test-postgres pg_isready; do sleep 1; done' >/dev/null 2>&1
    fi
    
    if ! docker ps --format "{{.Names}}" | grep -q "test-redis"; then
        docker run -d --name test-redis \
            -p 6380:6379 \
            redis:7-alpine >/dev/null 2>&1
    fi
}

# Test-Services starten
start_test_services() {
    log_debug "Starte Test-Services..."
    
    # Test-Compose Datei falls vorhanden
    local test_compose="$REPO_ROOT/docker-compose.test.yml"
    if [[ -f "$test_compose" ]]; then
        docker compose -f "$test_compose" up -d >/dev/null 2>&1
    fi
    
    # Mock-Services für Tests
    start_mock_services
}

# Mock-Services für Tests
start_mock_services() {
    log_debug "Starte Mock-Services..."
    
    # Einfacher HTTP-Mock-Server
    if command -v python3 >/dev/null; then
        python3 -c "
import http.server
import socketserver
import threading
import json

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode())
        elif self.path.startswith('/api/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'mock': True, 'path': self.path}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        self.do_GET()
    
    def log_message(self, format, *args):
        pass

def start_mock_server():
    with socketserver.TCPServer(('localhost', 18000), MockHandler) as httpd:
        httpd.serve_forever()

server_thread = threading.Thread(target=start_mock_server, daemon=True)
server_thread.start()
print('Mock server started on port 18000')
" &
        
        export MOCK_SERVER_PID=$!
        sleep 1  # Kurze Wartezeit für Server-Start
    fi
}

# Unit Tests ausführen
run_unit_tests() {
    log_info "Führe Unit Tests aus..."
    
    cd "$REPO_ROOT" || return 1
    
    local pytest_args=()
    
    # Test-Markierungen
    pytest_args+=("-m" "not integration and not e2e and not slow")
    
    # Coverage falls aktiviert
    if [[ "$TEST_COVERAGE" == "true" ]]; then
        pytest_args+=("--cov=." "--cov-report=html:test-coverage/html" "--cov-report=xml:test-coverage/coverage.xml")
    fi
    
    # Parallelisierung
    if [[ "$TEST_PARALLEL" == "true" ]] && command -v pytest-xdist >/dev/null; then
        pytest_args+=("-n" "auto")
    fi
    
    # Verbose falls aktiviert
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v" "-s")
    else
        pytest_args+=("-q")
    fi
    
    # Output-Format
    pytest_args+=("--tb=short")
    
    # JUnit XML für CI
    pytest_args+=("--junit-xml=test-results/unit-tests.xml")
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/unit-tests.log; then
        TEST_RESULTS[passed]=$((${TEST_RESULTS[passed]} + $(grep -c "PASSED" test-results/unit-tests.log || echo "0")))
        log_ok "Unit Tests abgeschlossen"
    else
        exit_code=$?
        TEST_RESULTS[failed]=$((${TEST_RESULTS[failed]} + $(grep -c "FAILED" test-results/unit-tests.log || echo "0")))
        log_err "Unit Tests fehlgeschlagen"
    fi
    
    TEST_RESULTS[total]=$((${TEST_RESULTS[passed]} + ${TEST_RESULTS[failed]}))
    return $exit_code
}

# Integration Tests ausführen
run_integration_tests() {
    log_info "Führe Integration Tests aus..."
    
    cd "$REPO_ROOT" || return 1
    
    local pytest_args=(
        "-m" "integration"
        "--tb=short"
        "--junit-xml=test-results/integration-tests.xml"
    )
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v" "-s")
    else
        pytest_args+=("-q")
    fi
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/integration-tests.log; then
        TEST_RESULTS[passed]=$((${TEST_RESULTS[passed]} + $(grep -c "PASSED" test-results/integration-tests.log || echo "0")))
        log_ok "Integration Tests abgeschlossen"
    else
        exit_code=$?
        TEST_RESULTS[failed]=$((${TEST_RESULTS[failed]} + $(grep -c "FAILED" test-results/integration-tests.log || echo "0")))
        log_err "Integration Tests fehlgeschlagen"
    fi
    
    return $exit_code
}

# E2E Tests ausführen
run_e2e_tests() {
    log_info "Führe End-to-End Tests aus..."
    
    # Prüfe ob Services laufen
    local required_services=(
        "http://localhost:3000:Frontend"
        "http://localhost:8000/health:API"
    )
    
    for service_info in "${required_services[@]}"; do
        local url="${service_info%%:*}"
        local name="${service_info##*:}"
        
        if ! curl -f -s --max-time 5 "$url" >/dev/null 2>&1; then
            log_err "E2E Tests: $name nicht erreichbar ($url)"
            return 1
        fi
    done
    
    cd "$REPO_ROOT" || return 1
    
    local pytest_args=(
        "-m" "e2e"
        "--tb=short"
        "--junit-xml=test-results/e2e-tests.xml"
    )
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v" "-s")
    else
        pytest_args+=("-q")
    fi
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/e2e-tests.log; then
        TEST_RESULTS[passed]=$((${TEST_RESULTS[passed]} + $(grep -c "PASSED" test-results/e2e-tests.log || echo "0")))
        log_ok "E2E Tests abgeschlossen"
    else
        exit_code=$?
        TEST_RESULTS[failed]=$((${TEST_RESULTS[failed]} + $(grep -c "FAILED" test-results/e2e-tests.log || echo "0")))
        log_err "E2E Tests fehlgeschlagen"
    fi
    
    return $exit_code
}

# API Tests ausführen
run_api_tests() {
    log_info "Führe API Tests aus..."
    
    # Prüfe API-Verfügbarkeit
    if ! curl -f -s --max-time 5 "http://localhost:8000/health" >/dev/null 2>&1; then
        log_warn "API nicht erreichbar - starte Mock-API für Tests"
        # Hier könnte eine Mock-API gestartet werden
    fi
    
    cd "$REPO_ROOT" || return 1
    
    # API Tests mit pytest
    local pytest_args=(
        "-m" "api"
        "--tb=short"
        "--junit-xml=test-results/api-tests.xml"
    )
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v" "-s")
    fi
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/api-tests.log; then
        log_ok "API Tests abgeschlossen"
    else
        exit_code=$?
        log_err "API Tests fehlgeschlagen"
    fi
    
    return $exit_code
}

# Frontend Tests ausführen
run_frontend_tests() {
    log_info "Führe Frontend Tests aus..."
    
    local frontend_dir="$REPO_ROOT/frontend/agent-ui"
    if [[ ! -d "$frontend_dir" ]]; then
        log_err "Frontend-Verzeichnis nicht gefunden"
        return 1
    fi
    
    cd "$frontend_dir" || return 1
    
    # npm test ausführen
    local exit_code=0
    if npm test -- --coverage --watchAll=false --testResultsProcessor=jest-junit 2>&1 | tee "$REPO_ROOT/test-results/frontend-tests.log"; then
        log_ok "Frontend Tests abgeschlossen"
    else
        exit_code=$?
        log_err "Frontend Tests fehlgeschlagen"
    fi
    
    # Test-Ergebnisse verschieben
    if [[ -f "junit.xml" ]]; then
        mv junit.xml "$REPO_ROOT/test-results/frontend-tests.xml"
    fi
    
    cd "$REPO_ROOT" || return 1
    return $exit_code
}

# MCP Tests ausführen
run_mcp_tests() {
    log_info "Führe MCP Tests aus..."
    
    # Prüfe MCP Services
    local mcp_services=(8001 8002 8003 8004 8005)
    local available_services=0
    
    for port in "${mcp_services[@]}"; do
        if curl -f -s --max-time 3 "http://localhost:$port/health" >/dev/null 2>&1; then
            available_services=$((available_services + 1))
        fi
    done
    
    if [[ $available_services -eq 0 ]]; then
        log_warn "Keine MCP Services verfügbar - Tests werden übersprungen"
        TEST_RESULTS[skipped]=$((${TEST_RESULTS[skipped]} + 1))
        return 0
    fi
    
    cd "$REPO_ROOT" || return 1
    
    local pytest_args=(
        "-m" "mcp"
        "--tb=short"
        "--junit-xml=test-results/mcp-tests.xml"
    )
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v" "-s")
    fi
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/mcp-tests.log; then
        log_ok "MCP Tests abgeschlossen"
    else
        exit_code=$?
        log_err "MCP Tests fehlgeschlagen"
    fi
    
    return $exit_code
}

# Performance Tests ausführen
run_performance_tests() {
    log_info "Führe Performance Tests aus..."
    
    cd "$REPO_ROOT" || return 1
    
    # Performance Tests mit pytest-benchmark
    local pytest_args=(
        "-m" "performance"
        "--benchmark-only"
        "--benchmark-json=test-results/benchmark.json"
        "--tb=short"
    )
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v")
    fi
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/performance-tests.log; then
        log_ok "Performance Tests abgeschlossen"
    else
        exit_code=$?
        log_err "Performance Tests fehlgeschlagen"
    fi
    
    return $exit_code
}

# Security Tests ausführen
run_security_tests() {
    log_info "Führe Security Tests aus..."
    
    # Bandit für Python Security
    if command -v bandit >/dev/null; then
        log_debug "Führe Bandit Security Scan aus..."
        bandit -r . -f json -o test-results/security-bandit.json 2>/dev/null || true
    fi
    
    # Safety für bekannte Vulnerabilities
    if command -v safety >/dev/null; then
        log_debug "Führe Safety Check aus..."
        safety check --json --output test-results/security-safety.json 2>/dev/null || true
    fi
    
    # Weitere Security Tests
    cd "$REPO_ROOT" || return 1
    
    local pytest_args=(
        "-m" "security"
        "--tb=short"
        "--junit-xml=test-results/security-tests.xml"
    )
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        pytest_args+=("-v" "-s")
    fi
    
    local exit_code=0
    if pytest "${pytest_args[@]}" tests/ 2>&1 | tee test-results/security-tests.log; then
        log_ok "Security Tests abgeschlossen"
    else
        exit_code=$?
        log_err "Security Tests fehlgeschlagen"
    fi
    
    return $exit_code
}

# Code Quality Checks
run_quality_checks() {
    log_info "Führe Code Quality Checks aus..."
    
    cd "$REPO_ROOT" || return 1
    
    local quality_exit_code=0
    
    # Ruff für Python Linting
    if command -v ruff >/dev/null; then
        log_debug "Führe Ruff Check aus..."
        if ruff check . --output-format=json --output-file=test-results/ruff.json; then
            log_ok "Ruff Check erfolgreich"
        else
            log_warn "Ruff gefunden Issues"
            quality_exit_code=1
        fi
    fi
    
    # MyPy für Type Checking
    if command -v mypy >/dev/null; then
        log_debug "Führe MyPy Check aus..."
        if mypy . --junit-xml test-results/mypy.xml 2>/dev/null; then
            log_ok "MyPy Check erfolgreich"
        else
            log_warn "MyPy gefunden Type Issues"
            quality_exit_code=1
        fi
    fi
    
    # Black für Code Formatting Check
    if command -v black >/dev/null; then
        log_debug "Führe Black Check aus..."
        if black --check --diff . 2>/dev/null; then
            log_ok "Black Formatting Check erfolgreich"
        else
            log_warn "Black gefunden Formatting Issues"
            quality_exit_code=1
        fi
    fi
    
    return $quality_exit_code
}

# Test-Ergebnisse zusammenfassen
generate_test_report() {
    log_info "Generiere Test-Report..."
    
    local report_file="$REPO_ROOT/test-results/test-report.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Agent-NN Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .stats { display: flex; gap: 20px; margin: 20px 0; }
        .stat { background: #e8f4f8; padding: 15px; border-radius: 5px; flex: 1; text-align: center; }
        .passed { background: #d4edda; }
        .failed { background: #f8d7da; }
        .skipped { background: #fff3cd; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Agent-NN Test Report</h1>
        <p>Generiert am: $(date)</p>
    </div>
    
    <div class="stats">
        <div class="stat passed">
            <h3>${TEST_RESULTS[passed]}</h3>
            <p>Bestanden</p>
        </div>
        <div class="stat failed">
            <h3>${TEST_RESULTS[failed]}</h3>
            <p>Fehlgeschlagen</p>
        </div>
        <div class="stat skipped">
            <h3>${TEST_RESULTS[skipped]}</h3>
            <p>Übersprungen</p>
        </div>
        <div class="stat">
            <h3>${TEST_RESULTS[total]}</h3>
            <p>Gesamt</p>
        </div>
    </div>
    
    <div class="section">
        <h2>Test-Dateien</h2>
        <ul>
EOF
    
    # Test-Dateien auflisten
    for file in test-results/*.xml test-results/*.json test-results/*.log; do
        if [[ -f "$file" ]]; then
            echo "            <li><a href=\"$(basename "$file")\">$(basename "$file")</a></li>" >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF
        </ul>
    </div>
    
    <div class="section">
        <h2>Coverage Report</h2>
EOF
    
    if [[ -f "test-coverage/coverage.xml" ]]; then
        echo "        <p><a href=\"../test-coverage/html/index.html\">HTML Coverage Report</a></p>" >> "$report_file"
    else
        echo "        <p>Keine Coverage-Daten verfügbar</p>" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
    </div>
</body>
</html>
EOF
    
    log_ok "Test-Report erstellt: $report_file"
}

# Test-Umgebung bereinigen
cleanup_test_environment() {
    if [[ "$CLEANUP_AFTER" == "false" ]]; then
        log_info "Cleanup übersprungen"
        return
    fi
    
    log_info "Bereinige Test-Umgebung..."
    
    # Test-Container stoppen
    local test_containers=("test-postgres" "test-redis")
    for container in "${test_containers[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "$container"; then
            docker stop "$container" >/dev/null 2>&1 || true
            docker rm "$container" >/dev/null 2>&1 || true
        fi
    done
    
    # Test-Compose Services stoppen
    local test_compose="$REPO_ROOT/docker-compose.test.yml"
    if [[ -f "$test_compose" ]]; then
        docker compose -f "$test_compose" down >/dev/null 2>&1 || true
    fi
    
    # Mock-Server stoppen
    if [[ -n "${MOCK_SERVER_PID:-}" ]]; then
        kill "$MOCK_SERVER_PID" 2>/dev/null || true
    fi
    
    log_debug "Test-Umgebung bereinigt"
}

# Test-Zusammenfassung anzeigen
show_test_summary() {
    echo
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                             TEST-ZUSAMMENFASSUNG                            ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    printf "║  Gesamt:       %3d Tests                                                     ║\n" "${TEST_RESULTS[total]}"
    printf "║  ✅ Bestanden:   %3d Tests                                                     ║\n" "${TEST_RESULTS[passed]}"
    printf "║  ❌ Fehlgeschlagen: %3d Tests                                                  ║\n" "${TEST_RESULTS[failed]}"
    printf "║  ⏭️ Übersprungen: %3d Tests                                                   ║\n" "${TEST_RESULTS[skipped]}"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    
    # Success Rate berechnen
    if [[ ${TEST_RESULTS[total]} -gt 0 ]]; then
        local success_rate=$((${TEST_RESULTS[passed]} * 100 / ${TEST_RESULTS[total]}))
        echo "Success Rate: ${success_rate}%"
    fi
    
    echo
    echo "📁 Test-Artefakte: test-results/"
    if [[ "$TEST_COVERAGE" == "true" ]]; then
        echo "📊 Coverage Report: test-coverage/html/index.html"
    fi
    
    # Empfehlungen bei Fehlern
    if [[ ${TEST_RESULTS[failed]} -gt 0 ]]; then
        echo
        echo "🔍 Bei Test-Fehlern:"
        echo "• Detaillierte Logs in test-results/*.log prüfen"
        echo "• Tests mit --verbose für mehr Details ausführen"
        echo "• Einzelne Tests mit --filter isoliert testen"
    fi
}

# Hauptfunktion
main() {
    local categories=()
    local test_filter=""
    local output_format="text"
    local test_timeout=300
    
    # Parameter parsen
    while [[ $# -gt 0 ]]; do
        case $1 in
            --coverage)
                TEST_COVERAGE=true
                shift
                ;;
            --verbose)
                TEST_VERBOSE=true
                shift
                ;;
            --quick)
                TEST_QUICK=true
                shift
                ;;
            --parallel)
                TEST_PARALLEL=true
                shift
                ;;
            --no-parallel)
                TEST_PARALLEL=false
                shift
                ;;
            --integration)
                TEST_INTEGRATION=true
                categories+=("integration")
                shift
                ;;
            --e2e)
                TEST_E2E=true
                categories+=("e2e")
                shift
                ;;
            --save-artifacts)
                SAVE_ARTIFACTS=true
                shift
                ;;
            --no-cleanup)
                CLEANUP_AFTER=false
                shift
                ;;
            --timeout)
                test_timeout="$2"
                shift 2
                ;;
            --output)
                output_format="$2"
                shift 2
                ;;
            --filter)
                test_filter="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            unit|integration|e2e|performance|security|api|frontend|mcp)
                categories+=("$1")
                shift
                ;;
            all)
                categories=(unit integration e2e api frontend mcp)
                TEST_INTEGRATION=true
                TEST_E2E=true
                shift
                ;;
            fast)
                categories=(unit)
                TEST_QUICK=true
                shift
                ;;
            full)
                categories=(unit integration e2e performance security api frontend mcp)
                TEST_INTEGRATION=true
                TEST_E2E=true
                shift
                ;;
            *)
                log_err "Unbekannte Option: $1"
                usage >&2
                exit 1
                ;;
        esac
    done
    
    # Standard-Kategorien falls keine angegeben
    if [[ ${#categories[@]} -eq 0 ]]; then
        categories=(unit)
    fi
    
    log_info "Starte Test-Suite für: ${categories[*]}"
    
    # Trap für Cleanup
    trap cleanup_test_environment EXIT
    
    # Test-Umgebung vorbereiten
    setup_test_environment
    
    # Tests ausführen
    local overall_exit_code=0
    
    for category in "${categories[@]}"; do
        case "$category" in
            unit)
                run_unit_tests || overall_exit_code=1
                ;;
            integration)
                run_integration_tests || overall_exit_code=1
                ;;
            e2e)
                run_e2e_tests || overall_exit_code=1
                ;;
            performance)
                run_performance_tests || overall_exit_code=1
                ;;
            security)
                run_security_tests || overall_exit_code=1
                ;;
            api)
                run_api_tests || overall_exit_code=1
                ;;
            frontend)
                run_frontend_tests || overall_exit_code=1
                ;;
            mcp)
                run_mcp_tests || overall_exit_code=1
                ;;
            *)
                log_warn "Unbekannte Test-Kategorie: $category"
                ;;
        esac
    done
    
    # Code Quality Checks (immer ausführen)
    run_quality_checks || true  # Nicht als Fehler werten
    
    # Test-Report generieren
    generate_test_report
    
    # Zusammenfassung anzeigen
    show_test_summary
    
    exit $overall_exit_code
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
