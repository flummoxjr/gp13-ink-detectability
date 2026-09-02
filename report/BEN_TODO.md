# What only Ben can do before this ships

> **2026-08-24: every in-document marker is resolved and removed.** The report, sections, community posts and reproducibility doc are self-contained — decisions taken: repo `github.com/flummoxjr/gp13-ink-detectability` (created, private until submission), license MIT, data CC-BY-NC 4.0, tiles.parquet shipped, Paris-4 upgrade framed as a community recommendation, LOG.md linked as provenance. **What actually remains, all logistics:** (1) push the packaged tree and flip the repo public; (2) join Discord, then fill the Google Form — every field is pre-written in REPRODUCIBILITY §H; (3) file Track B per `issue_drafts/FILING_CHECKLIST.md`; (4) post the three community drafts. Nothing scientific is open.

Deadline: **Aug 31, 2026, 11:59 pm Pacific**, via the Google Form linked at scrollprize.org/prizes.
Everything below is either your voice, your judgement call, or your account.

## A. Must do — the report cannot be submitted without these

| # | Item | Where | Why it's yours |
|---|---|---|---|
| A1 | **DONE 2026-08-23.** Ben confirmed the letters claim after independent re-verification of the figure: report copy hash-identical to the primary, AUC recomputed independently at 0.9991, all 13 label components audited against the supervision mask (the two empty contours in panel 3 lie wholly outside it and are excluded from scoring by design — now explained in the caption) | `figures/ink9um_w035_control.png` | Closed. |
| A2 | **Opening framing paragraph + AI-assistance disclosure** | REPORT.md, top | villa's contributing norms expect disclosure that AI agents were used; volunteered reads far better than discovered. **Written 2026-08-21** — kept deliberately plain: it states that AI agents were used under your direction and what you personally did, without editorialising about degree. Also states who you are and the one-line thesis. |
| A3 | **Motivation sentence per section** (4 of them) | 01 §intro, 02 §intro, 03 §intro + §3.6, 04 §intro | The judges' stated bar is that work arises from a human pursuing the project's goals. Each section deliberately stops short of speaking for you. |
| A4 | **Closing claim paragraphs** (2) | 02 end, 04 end | What you are actually claiming from the instrument section and the methodology section. Deliberately left unwritten — a claim in someone else's voice is the thing to avoid here. |
| A5 | **Report title** — pick one of the two offered or write your own | REPORT.md, top | |
| A6 | **License + public repo** — confirm MIT for code, CC-BY-NC 4.0 for derived data, push the repo, put the URL in the report | REPORT.md "Released artifacts" | Hard eligibility requirement; winning code must be public under a permissive license at submission time. |
| A7 | **Final closing paragraph** — what you want to do next and what you're offering the team | REPORT.md, end | The single highest-response part of a submission in my reading of past winners. |
| A8 | **PHerc0813 status — final (2026-08-23).** Gate withdrawn (patches are oblique to the sheets and can test nothing). The normal-grid hypothesis was tested twice — first attempt invalid and disclosed, corrected run negative: |n_z| 0.974 → 0.986, cause still open. Two candidates remain untested (`mode: seed` vs `random_seed`; `min_area_cm`), both parked until after submission. **Nothing here needs money or action before the deadline** — just read §2.9.1's closing bracket and pick how you want the open question framed | §2.9.1 the §2.9.1 closing status line | The report's only forward-looking claim now closes honestly instead of promising. |
| A9 | **Confirm the cloud-spend split** — the report now says ≈$11 ($5.61 core + ≈$5.5 corpus survey); the $80 RunPod cap is shared with a parallel session and account spend was $40.95 on 2026-08-18 | REPORT.md exec summary | Only you can see the account. B4 below turns on this number being right. |

## B. Judgement calls — I have a recommendation, you decide

| # | Decision | My recommendation |
|---|---|---|
| B1 | Is the Paris 4 5× sensitivity upgrade (§3.5) framed as *our* proposed follow-up work or as a community recommendation? | Community recommendation, with an offer to run it — you don't yet have beamtime-grade calibration and promising it is a liability. |
| B2 | Include the two optional personal lines (S1a v1+v2 dying; "negative screen, positive methodology")? | Include both, briefly. The corrections ledger is the differentiator and a human voice on it lands. |
| B3 | Submit the whole portfolio as one entry, or split (index / instrument+battery / QC methodology)? | One entry. The through-line — *measure before you hunt* — is stronger than any piece alone, and splitting invites each piece to be judged as small. |
| B4 | Do we mention the running cost (≈$11 total: $5.61 core + ≈$5.5 corpus survey) prominently? | Yes. "Screening every published Grand-Prize segment cost $5.5" is a memorable, verifiable claim about method efficiency and cuts against the assumption that diagnostics require a cluster. |
| B5 | §2.7 (the scope correction) sits immediately after the corpus result and materially narrows it. Keep it there, or move it to Limitations? | Keep it there. A negative that states its own bound in the same breath is far harder to attack than one that buries it, and the 23-debug-dumps / 18-outside-the-volume facts are findings in their own right. |
| B6 | §2.6 (the refuted flag) — keep the full refutation, or compress to a line? | Keep it. It is the strongest single piece of evidence that the screen self-polices, and the four fixes it produced are reusable by anyone screening prediction maps. |

## C. Track B (separate from the report, and time-sensitive)

| # | Item | Status |
|---|---|---|
| C1 | **torch.compile / TritonMissing issue** — needs the predict-side traceback pasted in, then your commentary | Draft ready at `trackD/issue_drafts/torch_compile_windows.md`; complements your own PR #1480. Reproducing the traceback takes ~2 min on any pod or your box. |
| C2 | **finalize_outputs NameError** — BLOCKED as duplicate of axiosdevs' PR #1430 | Do not file. A support comment is drafted in the same folder — your PHerc1203 encounter is exactly the real-usage evidence that PR lacks. Post it in your words. |
| C3 | Re-run the duplicate searches at filing time | The tracker moves ~24 items/day; both drafts note this. |

## D. Status of verification (so you know what's checked and what isn't)

- **Numbers**: the headline figures in REPORT.md and the four sections were checked against their primary artifacts (`out/k2b_index/*.json`, `out/ink9um_w035_scores.json`, `out/ink9um_1203_stats.json`, `out/k3_s2_stats.json`, `out/k1b_depth_validation.json`, the verdict/QC/comb documents). Every one I checked matched.
- **Not done**: the planned independent line-by-line fact-check pass and the external prize-fit review both died when the monthly spend limit hit mid-workflow. If you want them, they're one workflow re-run away and worth it — a wrong number in a report whose whole pitch is rigor is the worst possible own-goal.
- **Separability axis + corpus screen v2 (2026-08-18, late)**: §1.8, §2.4 and §2.6 were rewritten, and `verify_report.py` grew from 10 to **14 checks** — it now asserts the v2 corpus numbers (it had been validating superseded v1 ones), the full separability axis, the mesh-alignment forensics, and a sweep that **fails the build** if a superseded v1 corpus number reappears without being marked as history. Corrections ledger rows 12–15 were added, four of them against results in this report. 0 problems as of this writing.
- **Corpus survey integration (2026-08-18)**: §2.4–2.9 were rewritten around the 80/80 corpus screen. Every number in them is recomputed from primary files by `report/scripts/corpus_summary.py`, and `report/scripts/verify_report.py` now asserts the headline corpus and PHerc0813 figures against `out/survey/survey_all.json`, `out/survey/corpus_analysis.json` and `hunt/pherc0813_mesh_qc.json` — it reports 0 problems as of this writing. Re-run it after any edit to §2.
- **Fixed during integration**: four cross-section inconsistencies (a duplicated index table in §4.5 that carried a superseded "three degraded scrolls" count and a wrong claim about the PHerc0139 residual-reference bias direction; a duplicated fleet-battery table; a dangling §[X] cross-reference; and broken figure paths in all four sections that would have rendered every image as a dead link in the published repo).

## E. Suggested order of work

1. A1 (5 min — look at the figure).
2. A2 + A3 + A4 + A7 in one sitting (~60–90 min; these are all "your voice" and flow better written together).
3. A5, A6, B1–B4 (~20 min of decisions).
4. C1 + C2 (~30 min, and worth doing *before* the report — the tracker is fast and an accepted fix strengthens the submission's credibility).
5. Optional: re-run the fact-check + prize-fit reviewers.
