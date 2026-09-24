# SocialOmni leaderboard

Static English / Chinese results page for GitHub Pages. No build step or third-party JavaScript dependencies. Serve this directory over HTTP for local preview; the page loads `data.json` with `fetch`.

Only the `modern-20260924` judge panel is displayed. The paper-answer and hosted-run groups share that panel but retain separate rankings. Previous panels and the original source materials remain in the [repository archive](https://github.com/MAC-AutoML/SocialOmni/tree/main/evaluation/results/archive).

## Results data

`data.json` contains:

- `updated_at`: ISO calendar date, or `null` before publication.
- `links`: public `paper`, `code` and `dataset` URLs.
- `cohorts`: independent evaluation groups. Each has `id`, bilingual `label` and `description`, `source: {label, url}`, `samples: {who, when, gold_positive}`, and optional bilingual `notes`.
- `records`: each has `id`, `cohort`, `model`, optional `variant`, and `metrics: {who, when, qgold, qens, cov_plus, qens_joint}`. Optional `notes` and `source` identify row-specific qualifications. `status: "pending"` suppresses all scores. `status: "partial"` labels incomplete quality scoring; completed classification metrics may be shown, with missing quality values kept null.

Bilingual fields use `{"en": "English text", "zh": "中文文本"}`. Source labels may also be plain strings. Scores are numbers on a 0–100 scale. Use `null` for missing or disputed values; never use zero as a placeholder. Unknown sample counts are `null`. Records are displayed only inside their stated group; no overall score or cross-group ranking is calculated. Sorting always places missing values last.

Before publishing a result, verify its source, model identity, prompt and judge configuration, dataset version and sample selection. Separate paper transcriptions from independently run evaluations unless protocol equivalence has been established. Keep private manuscripts, credentials and internal service addresses out of this directory.

## Checks

Verify English and Chinese, metric sorting in both directions, search with no matches, an empty group, a failed data request, keyboard navigation, and horizontal table scrolling. Check at 320, 375, 414, 768 and desktop widths. The page should have no horizontal overflow outside the table region.
