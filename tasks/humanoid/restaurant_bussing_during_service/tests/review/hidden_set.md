# The fixed hidden set (maintainers' record; nothing runs this file)

`tests/hidden/seeds.json` holds the three cell seeds of the graded shifts. They were drawn once, before any of them was
run or looked at, by the rule written into the file itself:

    seed_k = 1000 + int(sha256("restaurant_bussing_during_service hidden set v1/k")[:12], 16) mod (2^31 - 1000),
    k = 0, 1, 2

skipping the development seeds (1-5, 11, 12, 101, 102: the seeds the cell, the reference busser and the survey photos
were developed on). No seed was replaced on its outcome. Agents and the oracle get the same three shifts.

The anchor (`tests/anchor.json`) is the privileged reference busser's own measured score on these three shifts.
