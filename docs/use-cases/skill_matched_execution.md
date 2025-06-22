# Skill matched execution

Certain tasks require specialised knowledge. By declaring required skills when
submitting a task, the dispatcher only selects agents with valid certifications.

Example: medical text analysis requires the `medical_reader` skill. Agents
without the certificate receive a `missing_skills` warning and are skipped when
`enforce_certification` is enabled.

