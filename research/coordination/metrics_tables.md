| PR | issue | opened | merged | open->merge (h) | critic blocking rounds (visible+hidden) | items/round | fix latency min (critic->builder) | critic latency min (builder->critic) | tests trajectory | clear->merge s |
|---|---|---|---|---|---|---|---|---|---|---|
| #1 | #3 | 10-02T18:24Z | 10-02T19:16 | 0.87 | 3+0 | [4, 2, 1] | [6.1, 13.8, 8.2] | [14.1, 5.7, 2.3, 1.8] | [] | 10 |
| #2 | #5 | 10-02T18:25Z | 10-02T19:16 | 0.86 | 3+0 | [5, 3, 1] | [6.3, 13.8, 8.2] | [13.7, 5.5, 2.3, 1.7] | [] | 22 |
| #21 | #4 | 10-02T19:45Z | 10-02T21:01 | 1.27 | 2+0 | [3, 2] | [16.0, 9.9] | [25.1, 8.3, 17.1] | [61, 77, 85] | 3 |
| #23 | #7 | 10-02T21:33Z | 10-03T16:40 | 19.11 | 2+0 | [4, 3] | [61.2, 357.0] | [19.0, 705.0, 4.0] | [127, 137, 143] | 3 |
| #24 | #8 | 10-03T17:14Z | 10-04T10:39 | 17.43 | 3+0 | [4, 3, 2] | [14.2, 16.8, 289.1] | [10.2, 30.4, 15.3, 669.6] | [178, 188, 194, 199] | 3 |
| #25 | #9 | 10-04T17:53Z | 10-07T16:21 | 70.46 | 2+0 | [6, 3] | [1062.3, 29.6] | [157.2, 8.0, 2970.6] | [236, 243, 248] | 3 |
| #26 | #10 | 10-07T16:48Z | 10-08T12:33 | 19.75 | 2+1 | [3, 1] | [19.1, 6.3] | [1123.6, 6.3, 2.8] | [283, 298, 307, 312] | 9 |
| #27 | #11 | 10-08T13:03Z | 10-12T14:20 | 97.28 | 3+0 | [6, 3, 2] | [417.6, 648.6, 3011.6] | [8.5, 213.8, 1534.3, 2.4] | [348, 360, 365, 368] | 8 |
| #28 | #12 | 10-12T15:47Z |  | None | 5+0 | [8, 10, 9, 5, 2] | [14.6, 122.8, 147.3, 14.3] | [21.4, 3.2, 17.9, 3.7, 2.4] | [406, 406, 405, 428, 438] | None |

OVERALL
{
 "fix_latency_min": {
  "n": 24,
  "median": 16.4,
  "mean": 263.1,
  "min": 6.1,
  "max": 3011.6,
  "p_le_30min": 0.62,
  "over_8h": 3
 },
 "critic_latency_min": {
  "n": 33,
  "median": 10.2,
  "mean": 231.1,
  "min": 1.7,
  "max": 2970.6,
  "p_le_30min": 0.76,
  "over_8h": 5
 },
 "open_to_merge_h": {
  "n": 8,
  "median": 18.3,
  "mean": 28.4,
  "min": 0.86,
  "max": 97.28,
  "p_le_30min": 0.75,
  "over_8h": 0
 },
 "total_visible_blocking_rounds_merged": 20,
 "total_blocking_rounds_incl_hidden_merged": 21,
 "merged_prs": 8,
 "merged_with_zero_blocking_rounds": 0,
 "clear_to_merge_s": {
  "1": 10,
  "2": 22,
  "21": 3,
  "23": 3,
  "24": 3,
  "25": 3,
  "26": 9,
  "27": 8,
  "28": null
 },
 "category_totals": {
  "schema/contract validation": 50,
  "determinism/reproducibility": 40,
  "provenance/audit evidence": 29,
  "hashing/identity": 24,
  "immutability/aliasing": 22,
  "pre-reg/protocol constants": 17,
  "atomicity/commit": 8,
  "leakage/oracle labels": 4,
  "type strictness": 4,
  "lifecycle/state machine": 3
 },
 "ledger_rows": 173,
 "rows_by_thread_type": {
  "issue_other": 63,
  "pr": 56,
  "issue#20": 54
 },
 "note": "categories are regex-heuristic over item text (see report for the manual categorisation); PR1/PR2 are doc/ADR PRs and included.",
 "issue20_share": 0.312,
 "posted_to_3_threads_per_event": true
}
