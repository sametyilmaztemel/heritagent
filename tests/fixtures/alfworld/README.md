# ALFWorld game fixture

`game.tw-pddl` is one unmodified ALFWorld game, used by
`tests/runtime/test_alfworld_adapter.py` to open and step a real game when
`ALFWORLD_GAMEFILE` is not set (ADR-0003 D8).

- Source: ALFWorld release 0.2.2, archive `json_2.1.1_tw-pddl.zip`
  (sha256 `d376c4bbb097c068528a2a9805e358e4cd14bc75c4d68a24e5141aebd234ef67`,
  the archive `evaluation/exp0001/data.py` downloads), member
  `json_2.1.1/valid_seen/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-406/trial_T20190908_122858_883968/game.tw-pddl`.
- File sha256: `9561438d283b0b50ed4e58dfe26cc0e1b6ad7e0dce89ce4fe8f9e2b72aeee50c`.
- It comes from `valid_seen`, a split EXP-0001 draws no pool from, and
  it is marked solvable, as the split loader requires.
- Licence: MIT, copyright (c) 2020 Mohit Shridhar and others. The licence
  text is in `LICENSE` next to this file.
