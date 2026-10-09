# Cross-check of the pv1 reference (de)

Until 2026-10-01 the German reference was a single Opus judge (`claude-a`), and the
only check on it was a second Opus judge (`claude-b`). Two judges of the same
family can share a blind spot. This cross-check adds two judges from other
Claude models, has every disagreement decided by a separate researched
adjudicator, and scores all arms again against the result (`adjudicated-a`).
The English lines (`claude-c`) were not cross-checked.

## Method

1. **Two more blind judges.** On 2026-10-02, 16:04 to 16:11 UTC, two agents
   judged all 636 German (line, question) pairs from the same judge input as
   claude-a (`.cache/judge-input/compact.de.json`), with web research allowed:
   `sonnet-a` (model `claude-sonnet-5-5`) and `fable-a` (model
   `claude-fable-5-1`). Files: `reference/pv1/sonnet-a.de.jsonl`,
   `reference/pv1/fable-a.de.jsonl`.
2. **Diff.** Every (line, question) where Opus (claude-a), Sonnet and Fable do
   not all agree: 61 pairs.
3. **Adjudication.** 2026-10-02, 16:11 to 16:46 UTC. Each of the 61 pairs went
   to its own fresh agent of one of the two other families, alternating:
   30 to `claude-sonnet-5-5`, 31 to `claude-fable-5-1`. The adjudicator saw the
   line, the posting, the CV, the three judge rows with their notes, the
   question wording in `bench/match.ts` and `bench/judge.html`, and was told to
   research on the web where a fact about a tool or method decides the case.
   Each returned a choice, a confidence (high, medium, low), a reason and its
   sources.
4. **Blindness.** No judge and no adjudicator opened a model result: the tool
   calls of both judge agents and all 62 adjudicator agents (one pair was
   started twice) contain no read of `results/pv1/<arm>/`, `score-*`,
   `significance-*` or `report.md`. Model IDs and times are taken from the
   workflow agent transcripts.
5. **Reference.** `reference/pv1/adjudicated-a.de.jsonl` is claude-a with the
   61 verdicts applied: new `choice`, `note` = verdict reason plus sources,
   `"adjudicated": true` and `"confidence"`. All other rows are claude-a
   unchanged. No verdict concerned is_req (all three judges agree on all 136
   lines), so no line changed from no to yes and no must, axis, evidence or
   level rows had to be added.

Still no human judgment.

## Agreement of the three judges

From `judges-agreement-de.md`, share of (line, question) pairs:

| Question | n | Opus-Sonnet | Opus-Fable | Sonnet-Fable | all three |
|---|---|---|---|---|---|
| is_req | 136 | 100.0% | 100.0% | 100.0% | 100.0% |
| must | 116 | 100.0% | 100.0% | 100.0% | 100.0% |
| axis | 116 | 98.3% | 94.0% | 92.2% | 92.2% |
| evidence | 116 | 85.3% | 94.0% | 82.8% | 81.0% |
| level | 116 | 90.5% | 85.3% | 86.2% | 81.0% |
| role | 4 | 100.0% | 75.0% | 75.0% | 75.0% |
| demand (8 areas) | 32 | 84.4% | 84.4% | 87.5% | 78.1% |
| all | 636 | 94.5% | 94.2% | 92.1% | 90.4% |

## Disagreements and verdicts

| Question | disagreements | verdict changed the Opus choice |
|---|---|---|
| is_req | 0 | 0 |
| must | 0 | 0 |
| axis | 9 | 2 |
| evidence | 22 | 7 |
| level | 22 | 13 |
| role | 1 | 0 |
| demand:testauto | 1 | 1 |
| demand:qm | 2 | 1 |
| demand:ai | 1 | 1 |
| demand:dev | 1 | 0 |
| demand:lead | 2 | 1 |
| **all** | **61** | **26** |

Confidence of the 61 verdicts: 16 high, 43 medium, 2 low.

Direction of the changes:

- **level:** all 13 changes raise the Opus level by one step (mostly 0 to 1
  where the CV shows adjacent work, and 1 to 2 where it shows the activity in
  one role). The adjudicated reference is one step more generous than Opus on
  these lines, never stricter.
- **evidence:** 5 of 7 changes replace an Opus entry with `none`, 2 replace it
  with another entry (p04-de-l07 `s-claims-agents`, p07-de-l08
  `s-s4-retail-testmgr`). Lines without evidence by the reference rise from 43
  to 48. The adjudicators read the evidence question strictly ("which entry
  shows that the candidate meets the requirement", none = "no entry shows
  it"): adjacency belongs in level, not in evidence. Sonnet named an entry on
  all 23 of its level-1 lines, which is the main source of evidence
  disagreements; only 1 of its 15 evidence dissents was upheld.
- **axis:** p03-de-l07 goes from ai to testauto and p10-de-l11 from dev to
  other. The same line in p01 and p02 stays ai, so the reference now answers
  the same posting line differently in p01/p02 and p03. Both verdicts are
  medium and flag the split.
- **project:** demand:ai of portal-testautomation 3 to 2, demand:lead of
  portal-testautomation 2 to 1, demand:qm of agent-platform 1 to 0,
  demand:testauto of agent-platform 0 to 1.

Who won, by lone dissenter (two judges agree, one differs; all 61 cases are of
this kind) and adjudicator family:

| lone dissenter | adjudicated by Sonnet: dissenter upheld | adjudicated by Fable: dissenter upheld | total upheld |
|---|---|---|---|
| Opus | 2 of 6 | 0 of 5 | 2 of 11 |
| Sonnet | 2 of 11 | 1 of 13 | 3 of 24 |
| Fable | 6 of 13 | 8 of 13 | 14 of 26 |

Fable's lone positions win about half the time with either adjudicator family,
Sonnet's rarely with either. There is no clear sign that an adjudicator
favours its own family.

## Judges against the adjudicated reference

| judge | axis | evidence | level | invented evidence | missed evidence | demand |
|---|---|---|---|---|---|---|
| claude-a (Opus) | 98.3 % | 94.0 % | 88.8 % | 10.4 % | 0.0 % | 87.5 % |
| claude-b (Opus, never adjudicated) | 95.7 % | 95.7 % | 92.2 % | 6.3 % | 1.5 % | 93.8 % |
| sonnet-a | 96.6 % | 84.5 % | 89.7 % | 33.3 % | 0.0 % | 90.6 % |
| fable-a | 95.7 % | 98.3 % | 94.8 % | 0.0 % | 1.5 % | 90.6 % |

claude-b took no part in the cross-check and still agrees with the adjudicated
reference slightly more than claude-a does on evidence and level. Its 94.8 %
on evidence against claude-a, quoted as the judge ceiling so far, is 95.7 %
against the adjudicated reference.

## Model scores: claude-a against adjudicated-a

Run 1 of each arm, German lines. is_req, must and role do not change (no
verdict touched them). Each cell: against `claude-a` / against `adjudicated-a`.

| arm | axis | evidence | level | level ±1 | invented evidence ↓ | missed evidence ↓ | demand |
|---|---|---|---|---|---|---|---|
| jev | 93.1 / 93.1 | 81.9 / 81.0 | 73.3 / 79.3 | 97.4 / 98.3 | 23.3 / 25.0 | 5.5 / 1.5 | 71.9 / 75.0 |
| winnow-12b | 86.2 / 87.1 | 86.2 / 85.3 | 71.6 / 77.6 | 98.3 / 98.3 | 16.3 / 18.8 | 5.5 / 1.5 | 71.9 / 75.0 |
| qwen3.8-27b | 86.2 / 86.2 | 85.3 / **87.9** | 62.9 / 69.0 | 100.0 / 100.0 | 9.3 / 10.4 | 8.2 / 2.9 | 71.9 / 75.0 |
| qwen3-32b | 75.9 / 76.7 | 79.3 / 77.6 | 41.4 / 47.4 | 92.2 / 95.7 | 30.2 / 33.3 | 2.7 / 0.0 | 71.9 / 75.0 |
| shisa-de-1 | 85.3 / 85.3 | 79.3 / 79.3 | 61.2 / 66.4 | 93.1 / 95.7 | 32.6 / 35.4 | 4.1 / 1.5 | 59.4 / 53.1 |

Unchanged: is_req jev 88.2, winnow-12b 86.8, the others 85.3; must jev 93.1,
winnow-12b 94.0, qwen3.8-27b 97.4, qwen3-32b 100.0, shisa-de-1 75.9.
Invented evidence is out of 43 lines against claude-a and 48 against
adjudicated-a; missed evidence out of 73 and 68.

Paired McNemar against Jev that change their verdict at p < 0.05
(`significance-de.md` against `significance-de-adjudicated.md`):

| question | arm | against claude-a | against adjudicated-a |
|---|---|---|---|
| axis | winnow-12b | 9 vs 1, p = 0.021 | 9 vs 2, p = 0.065 |

Every other p < 0.05 stays below 0.05 and every other p ≥ 0.05 stays above.
Notable shifts below the line: invented evidence qwen3.8-27b 0 vs 6,
p = 0.031 becomes 0 vs 7, p = 0.016; evidence qwen3.8-27b 6 vs 10, p = 0.454
becomes 4 vs 12, p = 0.077.

Files: `score-de-adjudicated.md` (all arms plus claude-a as comparison row),
`significance-de-adjudicated.md`.

## What happens to the article's conclusions

Article: https://bks-lab.com/blog/jev-gegen-lokale-modelle/ (bks-lab/bks-web, `src/content/blog/jev-gegen-lokale-modelle.mdx`),
German numbers against claude-a.

| Conclusion | Against adjudicated-a | Status |
|---|---|---|
| Winnow-12B is close to Jev | Agreement with Jev (88.6 %) does not depend on the reference. Level 77.6 vs Jev 79.3, p = 0.824; evidence 85.3 vs 81.0, p = 0.267 | holds |
| Winnow-12B is best on evidence | Qwen3.8-27B 87.9 % is now ahead of Winnow-12B 85.3 %; neither difference is significant | **changes** |
| Winnow-12B is significantly behind Jev on axis (9 vs 1, p = 0.021) | 9 vs 2, p = 0.065, no longer significant. Jev stays ahead on axis and stays significantly ahead of the other three local models | **changes** |
| Qwen3.8-27B invents evidence least | 10.4 % (5 of 48) against Jev 25.0 % (12 of 48), 0 vs 7, p = 0.016, stronger than before. The article's counts 10 and 4 of 43 become 12 and 5 of 48 | holds, numbers change |
| Qwen3.8-27B misses existing evidence somewhat more often | 2 of 68 against Jev 1 of 68 (was 6 vs 4 of 73); the difference nearly disappears | weakens |
| Qwen3.8-27B hits the level clearly less often than Jev | 69.0 vs 79.3, p = 0.036 | holds |
| All models invent evidence | lowest is Qwen3.8-27B at 10.4 %, highest Shisa DE-1 at 35.4 % | holds |
| Only the second judge never invents evidence | That was claude-b against claude-a. Against adjudicated-a, claude-a names an entry on 5 of 48 lines without evidence, claude-b on 3, fable-a on 0, sonnet-a on 16. The point that a judge row is not a ceiling holds | **changes** in wording |
| is_req disagreements are title and frame lines | is_req is untouched: Opus, Sonnet and Fable agree on all 136 lines, so three model families call the title and frame lines no | holds, stronger |
| Jev is best on level | Jev 79.3 % is still first, Winnow-12B 77.6 % second; all levels rise by 5 to 6 points because 13 reference levels went up by one | holds |

The English numbers in the article are against claude-c and were not
cross-checked.

## Limits

- The 61 verdicts are from models, not people. 43 of them are medium and 2 low
  confidence; most of the disputes are definition questions (how strict is
  "evidence", what counts as "adjacent", which area a data-engineering line
  belongs to), not facts.
- The strict evidence reading was applied only where the judges disagreed.
  Rows on which all three judges named an entry at level 1 stay as they were:
  4 level-1 rows in adjudicated-a name an entry, 3 of them on purpose by an
  adjudicator (the activity is shown, only weaker), 1 unchecked (p02-de-l06).
  Several adjudicators note that a bench-wide rule should be written down
  before a human sample.
- Two axis verdicts make the reference inconsistent across pairs on the same
  posting line (p01/p02-de-l07 ai, p03-de-l07 testauto).
- English is still a single Opus judge.
