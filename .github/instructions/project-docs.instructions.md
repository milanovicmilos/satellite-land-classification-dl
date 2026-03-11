---
description: "Use when creating or editing project documentation, specifications, README files, experiment notes, architecture docs, or planning files for this repository. Covers English technical writing, teammate responsibilities, reproducibility notes, and documentation boundaries."
name: "Project Documentation Rules"
applyTo: "**/*.md, **/*.txt, **/*.yml, **/*.yaml, **/*.json"
---

# Project Documentation Rules

## Documentation Language
- New technical documentation should be written in English.
- Keep code comments and configuration comments in English.
- Preserve existing Serbian academic material unless the user explicitly asks to translate or rewrite it.

## Required Project Context
- State that the project is EuroSAT land-use classification in Python.
- Reflect the agreed evaluation setup when relevant: stratified split, fixed seed, accuracy, macro F1-score, confusion matrix, precision, and recall.
- Distinguish shared infrastructure from model-specific work.

## Team Responsibilities
- Miloš Milanović: EfficientNetB0 and shared infrastructure.
- Bojan Živanić: ResNet50.
- Do not document Miloš as the implementer of both advanced models unless the user explicitly changes that rule.

## Documentation Quality
- Prefer concise, reproducible steps over vague guidance.
- When commands are not verified, label them as proposed or example commands.
- Document architectural decisions with clear boundaries between domain, application, infrastructure, and entrypoints.
- Prefer checklists, file trees, and concrete examples when they reduce ambiguity.