{{- define "agent-nn.name" -}}
{{ include "chart.name" . }}
{{- end -}}

{{- define "agent-nn.fullname" -}}
{{ include "chart.name" . }}-{{ .Release.Name }}
{{- end -}}
