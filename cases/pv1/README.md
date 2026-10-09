# Test cases pv1

Frozen test cases for the Jev match bench, version pv1. A service matches a freelancer CV against a project posting. For every posting line a model answers: is it a requirement, is it mandatory, which of 8 areas does it belong to, which CV entry is the evidence, and how well is it covered (level 0 to 3).

## Contents

- `postings/<id>.en.md` and `<id>.de.md`: four project postings in two languages. `portal-testautomation`, `agent-platform` and `sap-testmanagement` are copied verbatim from the demo postings of the site (`src/lib/match-demos.ts`). `data-platform` is new: a lakehouse data engineer posting that mostly falls outside the 8 areas, so the "other" area is exercised. It also contains lines that are not requirements (company sentence, benefits, how to apply).
- `cvs/<cv-id>.en.json` and `.de.json`: eight anonymous freelancer CVs in the `CvParams` shape of `match.ts` (profile.tagline, summary, education, language, experiences, projects). Every experience and project has a unique kebab anchor, identical in de and en. The de and en files are translations of each other. `cv06-junior-qa` is deliberately very short (2 stations, terse), `cv02-qa-lead-sap` is the long one (7 stations, 4 projects).
- `pairs.yaml`: the 10 posting and CV pairs to run.

## Everything is invented

All postings, companies, people, projects and numbers are made up. No real company or person appears, and the CVs carry no names.

## Freeze rule

pv1 is frozen since its reference judgment started (2026-10-01). Since then, no text, anchor or pair may change. Any change creates a new version (`cases/pv2/`) with a fresh reference judgment.

`design_intent` in `pairs.yaml` is the author's intent only. It is not the reference judgment. The reference is judged blind, line by line, in `reference/pv1/`: so far by model judges (see the main README), a human sample is still pending.

## Pairs

| id | posting | cv | design_intent |
|---|---|---|---|
| p01 | portal-testautomation | cv01-senior-testauto | fit |
| p02 | portal-testautomation | cv08-career-changer | partial |
| p03 | portal-testautomation | cv04-backend-jvm | no_fit |
| p04 | agent-platform | cv03-llm-agent-engineer | fit |
| p05 | agent-platform | cv01-senior-testauto | partial |
| p06 | agent-platform | cv06-junior-qa | no_fit |
| p07 | sap-testmanagement | cv02-qa-lead-sap | fit |
| p08 | sap-testmanagement | cv05-performance | partial |
| p09 | sap-testmanagement | cv03-llm-agent-engineer | no_fit |
| p10 | data-platform | cv07-data-engineer | fit |
