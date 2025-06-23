export default function MetricsPage() {
  return (
    <div className="p-4 h-full">
      <iframe
        src="/grafana/d/agentnn/agent-nn-overview?orgId=1&refresh=5s"
        className="w-full h-full border"
      />
    </div>
  )
}
