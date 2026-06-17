import pytest
from models.bayesian_network import (
    calculate_variance,
    apply_ebbinghaus_decay,
    calculate_expected_mastery,
    process_cognitive_update
)

def test_bayesian_update_with_time_decay():
    prior_alpha = 10.0
    prior_beta = 2.0
    time_delta_days = 30
    decay_rate = 0.05
    
    # Calculate decayed prior
    decayed_alpha = apply_ebbinghaus_decay(prior_alpha, time_delta_days, decay_rate)
    
    # Assert distribution flattens (uncertainty increases) over time
    assert decayed_alpha < prior_alpha
    assert calculate_variance(decayed_alpha, prior_beta) > calculate_variance(prior_alpha, prior_beta)

def test_expected_mastery():
    assert calculate_expected_mastery(1.0, 1.0) == 0.5
    assert calculate_expected_mastery(4.0, 1.0) == 0.8
    assert calculate_expected_mastery(0.0, 0.0) == 0.5

def test_cognitive_update_success():
    # A fresh user with 0 days elapsed
    alpha, beta, mastery, behavior_class = process_cognitive_update(
        prior_alpha=1.0,
        prior_beta=1.0,
        last_practiced_days=0,
        decay_rate=0.02,
        success=True,
        behavioral_flags=[]
    )
    assert alpha == 2.0
    assert beta == 1.0
    assert mastery == 2.0 / 3.0
    assert behavior_class == 0

def test_cognitive_update_failure():
    alpha, beta, mastery, behavior_class = process_cognitive_update(
        prior_alpha=1.0,
        prior_beta=1.0,
        last_practiced_days=0,
        decay_rate=0.02,
        success=False,
        behavioral_flags=[]
    )
    assert alpha == 1.0
    assert beta == 2.0
    assert mastery == 1.0 / 3.0
    assert behavior_class == 0

def test_copy_paste_dependency_handling():
    # Success attempt but user pasted code
    alpha, beta, mastery, behavior_class = process_cognitive_update(
        prior_alpha=1.0,
        prior_beta=1.0,
        last_practiced_days=0,
        decay_rate=0.02,
        success=True,
        behavioral_flags=[],
        telemetry_data={
            "paste_char_count": 50,
            "backspace_count": 0
        }
    )
    # behavior_class should be 2 (Copy-Paste)
    assert behavior_class == 2
    # alpha gets +0.1 (discounts 90%)
    assert alpha == 1.1
    assert beta == 1.0
    assert mastery == 1.1 / 2.1
