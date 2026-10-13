# Security policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems.
Report privately through [GitHub private vulnerability reporting](https://github.com/Unmesh30/firstread/security/advisories/new).

Include what you found, how to reproduce it, and the impact you expect. You should get an acknowledgement within 7 days.

## Scope

FirstRead is a non-commercial demo. In scope: the API (`services/api`), ingestion job, web app, and CI/CD configuration in this repository.
Out of scope: third-party services (Supabase, Upstash, Google Cloud, GitHub) themselves.

## Practices

- Secrets live only in `.env` (gitignored), GitHub Actions secrets and GCP Secret Manager.
- gitleaks runs in pre-commit and CI; Dependabot keeps dependencies current.
