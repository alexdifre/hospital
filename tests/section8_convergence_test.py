from tests.run_section8_experiments import (
    ExperimentResult,
    build_target_convergence_metrics,
    compute_target_convergence_trigger,
    derive_target_convergence_from_episodes,
)


def test_target_convergence_uses_max_non_mismatch_streak():
    assert compute_target_convergence_trigger(None, 6) == "non_mismatch_streak"


def test_saved_episode_bookkeeping_keeps_reached_streak_convergence():
    episodes = [
        {
            "episode": 3,
            "target_delta_norm": None,
            "mismatches": {"count": 1, "leg_count": 7},
            "target_convergence_legs": (
                [{"fuzzy_mismatch": False} for _ in range(6)]
                + [{"fuzzy_mismatch": True}]
            ),
        }
    ]

    bookkeeping = derive_target_convergence_from_episodes(episodes)

    assert bookkeeping["target_convergence_episode"] == 3
    assert bookkeeping["target_convergence_reason"] == "non_mismatch_streak"
    assert bookkeeping["target_final_non_mismatch_streak"] == 0
    assert bookkeeping["target_max_non_mismatch_streak"] == 6


def test_target_convergence_metrics_are_serializable_and_explicit():
    result = ExperimentResult(
        condition="full",
        profile="speed_oriented",
        seed=0,
        episodes=[
            {
                "target_delta_norm": 0.25,
                "terminal_target_updates": [
                    {"target_delta_norm": 0.1},
                    {"z_target_before": [1.0, 2.0], "z_target_after": [1.0, 2.3]},
                ],
            }
        ],
        mismatch_count=2,
        leg_count=10,
        mismatch_rate=0.2,
        target_convergence_episode=5,
        target_convergence_reason="non_mismatch_streak",
        target_final_delta_norm=0.25,
        target_min_delta_norm=0.25,
        target_final_non_mismatch_streak=1,
        target_max_non_mismatch_streak=6,
    )

    metrics = build_target_convergence_metrics(result)

    assert metrics["mpc_hyperparameter"] == "z_target"
    assert metrics["converged"] is True
    assert metrics["convergence_episode"] == 5
    assert metrics["updates"]["total"] == 2
    assert metrics["target_delta_norm"]["episode_stats"]["count"] == 1
    assert metrics["target_delta_norm"]["update_stats"]["count"] == 2
    assert metrics["non_mismatch_streak"]["max"] == 6
