from __future__ import annotations

from hypercircuit.utils.ids import member_key_from_event


def test_member_granularity_modes() -> None:
    ev = {
        "feature_space_id": "fs",
        "layer": -1,
        "node_type": "sae_features",
        "node_id": 7,
        "member_key": "fs:-1:sae_features:7",
    }
    assert member_key_from_event(ev, granularity="node_type") == "sae_features"
    assert member_key_from_event(ev, granularity="node_id") == "fs:-1:sae_features:7"
    ev_group = {**ev, "stable_node_id": "fs:-1:sae_features:7"}
    assert member_key_from_event(ev_group, granularity="group") == "fs:-1:sae_features:7"
    ev_group2 = {**ev, "stable_node_id": 99}
    assert member_key_from_event(ev_group2, granularity="group") == "fs:-1:sae_features:99"
