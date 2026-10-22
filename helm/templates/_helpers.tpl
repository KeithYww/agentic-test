{{/*
Expand the name of the chart.
*/}}
{{- define "agentic-qa.name" -}}
{{- .Chart.Name }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "agentic-qa.fullname" -}}
{{- .Release.Name }}
{{- end }}

{{/*
Common labels applied to all resources.
*/}}
{{- define "agentic-qa.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
