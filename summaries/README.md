# summaries/

Saved experiment results, organised by test phase.

## Structure

```
summaries/
  preliminary/       Early single-profile runs (safety_first) used to validate
  │                  the integration pipeline before adding the full learning stack.
  │                  Contains episode_00N.txt files alongside FINAL_SUMMARY.txt.
  │
  v4_profile_tests/  Per-profile runs with the v4 integrator (dual learning loops).
  │                  Used to verify each patient archetype converges correctly
  │                  before running the full multi-profile experiment suite.
  │                  Contains FINAL_SUMMARY.txt + learning_curves.csv per run.
  │
  full_system/       Full multi-task roster experiments (medication + meal preparation)
                     across multiple patient profiles. These are the results referenced
                     in the paper (Section 8).
                     Contains FINAL_SUMMARY.txt + learning_curves.csv per profile.
```

## Naming Convention

| Pattern | Meaning |
|---------|---------|
| `summaries_safety_first_YYYYMMDD_HHMMSS` | Preliminary run, safety_first profile |
| `summaries_v4_safety_first_YYYYMMDD_HHMMSS` | v4 integrator run, safety_first profile |
| `test_mixed_roster_<profile>` | Full system roster test for a given profile |
| `test_mixed_roster` | Full system roster test, default (uniform) profile |
