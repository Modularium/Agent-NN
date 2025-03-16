#!/bin/bash

# Farben für die Ausgabe
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Erstelle Agent-NN Dashboard Projektstruktur...${NC}"

# Hauptverzeichnisse erstellen
mkdir -p monitoring/dashboard/components/common
mkdir -p monitoring/dashboard/components/layout
mkdir -p monitoring/dashboard/components/panels
mkdir -p monitoring/dashboard/components/charts
mkdir -p monitoring/dashboard/context
mkdir -p monitoring/dashboard/hooks
mkdir -p monitoring/dashboard/utils
mkdir -p monitoring/dashboard/pages
mkdir -p monitoring/dashboard/types
mkdir -p monitoring/api/routes
mkdir -p data
mkdir -p logs

echo -e "${GREEN}Hauptverzeichnisse erstellt${NC}"

# Frontend-Dateien erstellen
# Common Components
touch monitoring/dashboard/components/common/Alert.tsx
touch monitoring/dashboard/components/common/Card.tsx
touch monitoring/dashboard/components/common/LoadingSpinner.tsx
touch monitoring/dashboard/components/common/ErrorBoundary.tsx
touch monitoring/dashboard/components/common/MetricCard.tsx
touch monitoring/dashboard/components/common/StatusBadge.tsx

# Layout Components
touch monitoring/dashboard/components/layout/Header.tsx
touch monitoring/dashboard/components/layout/Sidebar.tsx
touch monitoring/dashboard/components/layout/MainContent.tsx
touch monitoring/dashboard/components/layout/Footer.tsx

# Panel Components
touch monitoring/dashboard/components/panels/SystemOverviewPanel.tsx
touch monitoring/dashboard/components/panels/AgentsPanel.tsx
touch monitoring/dashboard/components/panels/ModelsPanel.tsx
touch monitoring/dashboard/components/panels/KnowledgePanel.tsx
touch monitoring/dashboard/components/panels/MonitoringPanel.tsx
touch monitoring/dashboard/components/panels/SecurityPanel.tsx
touch monitoring/dashboard/components/panels/TestingPanel.tsx
touch monitoring/dashboard/components/panels/SettingsPanel.tsx
touch monitoring/dashboard/components/panels/LogsPanel.tsx
touch monitoring/dashboard/components/panels/DocsPanel.tsx

# Chart Components
touch monitoring/dashboard/components/charts/BarChart.tsx
touch monitoring/dashboard/components/charts/LineChart.tsx
touch monitoring/dashboard/components/charts/PieChart.tsx
touch monitoring/dashboard/components/charts/MetricsChart.tsx

# Context
touch monitoring/dashboard/context/DashboardContext.tsx
touch monitoring/dashboard/context/ThemeContext.tsx
touch monitoring/dashboard/context/AuthContext.tsx

# Hooks
touch monitoring/dashboard/hooks/useSystemData.ts
touch monitoring/dashboard/hooks/useRefreshInterval.ts
touch monitoring/dashboard/hooks/useAgentData.ts
touch monitoring/dashboard/hooks/useMetrics.ts

# Utils
touch monitoring/dashboard/utils/api.ts
touch monitoring/dashboard/utils/formatters.ts
touch monitoring/dashboard/utils/chartHelpers.ts
touch monitoring/dashboard/utils/validators.ts

# Pages
touch monitoring/dashboard/pages/Dashboard.tsx

# Types
touch monitoring/dashboard/types/system.ts
touch monitoring/dashboard/types/agent.ts
touch monitoring/dashboard/types/model.ts
touch monitoring/dashboard/types/metrics.ts

# Main Frontend Files
touch monitoring/dashboard/App.tsx
touch monitoring/dashboard/index.tsx
touch monitoring/dashboard/index.css
touch monitoring/dashboard/package.json
touch monitoring/dashboard/tsconfig.json
touch monitoring/dashboard/tailwind.config.js

echo -e "${GREEN}Frontend-Dateien erstellt${NC}"

# Backend-Dateien erstellen
touch monitoring/api/server.py
touch monitoring/api/data_manager.py
touch monitoring/api/system_monitor.py
touch monitoring/api/__init__.py

# API Routes
touch monitoring/api/routes/__init__.py
touch monitoring/api/routes/system.py
touch monitoring/api/routes/agents.py
touch monitoring/api/routes/models.py
touch monitoring/api/routes/metrics.py

echo -e "${GREEN}Backend-Dateien erstellt${NC}"

# Docker-Dateien erstellen
touch docker-compose.yml
touch Dockerfile.api
touch Dockerfile.dashboard
touch requirements.txt

echo -e "${GREEN}Docker-Dateien erstellt${NC}"

# Erstelle eine einfache README.md
cat > README.md << 'EOF'
# Agent-NN Dashboard

Ein modernes Dashboard für die Überwachung und Verwaltung des Agent-NN Systems.

## Funktionen

- Systemübersicht
- Agent-Verwaltung
- Modell-Verwaltung
- Knowledge-Base-Verwaltung
- Monitoring
- Sicherheit
- A/B-Tests
- Einstellungen
- Logs und Alerts
- Dokumentation

## Installation

### Mit Docker

```bash
docker-compose up -d
```

### Manuelle Installation

Backend:
```bash
cd monitoring/api
pip install -r requirements.txt
uvicorn server:app --reload
```

Frontend:
```bash
cd monitoring/dashboard
npm install
npm start
```

## Zugriff

Dashboard: http://localhost:3000
API: http://localhost:8000

Demo-Zugangsdaten:
- Benutzername: admin
- Passwort: password
EOF

echo -e "${YELLOW}README.md erstellt${NC}"

echo -e "${BLUE}Erstellung der Projektstruktur abgeschlossen!${NC}"
echo -e "${YELLOW}Nächste Schritte:${NC}"
echo -e "1. Backend-Abhängigkeiten installieren: ${GREEN}pip install -r requirements.txt${NC}"
echo -e "2. Frontend-Abhängigkeiten installieren: ${GREEN}cd monitoring/dashboard && npm install${NC}"
echo -e "3. Mit Docker starten: ${GREEN}docker-compose up -d${NC}"
