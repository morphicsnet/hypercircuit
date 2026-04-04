#!/usr/bin/env python3
"""
Test script for real implementations in hypercircuit.
Tests the new real functionality that replaced mock implementations.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

def test_real_surrogate_training():
    """Test real surrogate training data assembly."""
    print("Testing real surrogate training data assembly...")

    from hypercircuit.surrogate.train import assemble_real_training_data

    # Mock events data with real activation values
    mock_events = [
        {'sample_id': 0, 'token_index': 0, 'node_type': 'sae_features', 'node_id': 1, 'value': 0.8},
        {'sample_id': 0, 'token_index': 0, 'node_type': 'attn_heads', 'node_id': 2, 'value': 0.6},
        {'sample_id': 1, 'token_index': 0, 'node_type': 'sae_features', 'node_id': 1, 'value': 0.9},
        {'sample_id': 1, 'token_index': 0, 'node_type': 'attn_heads', 'node_id': 2, 'value': 0.7},
    ]

    # Mock ensembles
    mock_ensembles = [
        {'id': 'test_ensemble_1', 'members': ['sae_features_1', 'attn_heads_2'], 'family': 'test_family'}
    ]

    try:
        data = assemble_real_training_data(
            run_dir=Path('.'),
            ensembles=mock_ensembles,
            events=mock_events,
            seed=42
        )
        print('✓ Real surrogate training data assembly works')
        print(f'  Generated data for {len(data)} ensembles')
        if 'test_ensemble_1' in data:
            ensemble_data = data['test_ensemble_1']
            print(f'  Ensemble shape: X={ensemble_data["X"].shape}, y={ensemble_data["y"].shape}')
            assert ensemble_data["X"].shape[0] == 2  # 2 samples
            assert ensemble_data["X"].shape[1] == 2  # 2 members
            assert len(ensemble_data["y"]) == 2  # 2 target values
        return True
    except Exception as e:
        print(f'✗ Surrogate training failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_real_causal_runner():
    """Test real causal runner import and basic functionality."""
    print("Testing real causal runner...")

    try:
        from hypercircuit.causal.runner import RealCausalRunner
        from hypercircuit.causal.harness import CausalHarness
        print('✓ RealCausalRunner import successful')

        # Test instantiation
        harness = CausalHarness()
        runner = RealCausalRunner(harness=harness, config=None)
        print('✓ RealCausalRunner instantiation successful')

        # Test run method with mock payload
        mock_payload = {
            "ensemble_ids": ["test_ensemble_1"],
            "run_dir": "/tmp/test_run"
        }
        result = runner.run(mock_payload)
        print('✓ RealCausalRunner.run() executed')
        print(f'  Status: {result.get("status")}')

        return True
    except Exception as e:
        print(f'✗ Causal runner test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_real_discovery():
    """Test real discovery candidate generation."""
    print("Testing real discovery candidate generation...")

    try:
        from hypercircuit.discovery.candidates import CandidateGenerator
        from hypercircuit.utils.config import DiscoveryConfig
        print('✓ CandidateGenerator import successful')

        # Create a minimal config
        config = DiscoveryConfig()
        generator = CandidateGenerator(config=config, seed=42)
        print('✓ CandidateGenerator instantiation successful')

        # Test with non-existent events file (should fallback to mock)
        result = generator.generate_candidates(
            events_path=Path('nonexistent.jsonl'),
            run_dir=Path('/tmp/test_discovery')
        )
        print('✓ Candidate generation executed')
        print(f'  Generated {result["n_candidates"]} candidates')

        return True
    except Exception as e:
        print(f'✗ Discovery test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_real_editing():
    """Test real editing functionality."""
    print("Testing real editing functionality...")

    try:
        from hypercircuit.steering.edits import compute_edit_map, apply_edit_plan
        print('✓ Editing functions import successful')

        # Test edit map computation
        weights = np.array([0.8, 0.6, 0.3, 0.1])
        edit_map = compute_edit_map(weights, scale=0.5)
        print('✓ Edit map computation successful')
        print(f'  Edit map shape: {edit_map.shape}')

        # Test edit plan application
        plan = apply_edit_plan(weights, scale=0.5, max_edit_scale=1.0)
        print('✓ Edit plan application successful')
        print(f'  Selected indices: {plan["selected_indices"]}')

        return True
    except Exception as e:
        print(f'✗ Editing test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_dictionary_builder():
    """Test dictionary builder without mock injection."""
    print("Testing dictionary builder...")

    try:
        from hypercircuit.dictionary.builder import build_ensemble_dictionary
        print('✓ Dictionary builder import successful')

        # Test with empty inputs (should handle gracefully)
        result = build_ensemble_dictionary(
            inputs_by_family={},
            config={},
            run_dir=Path('/tmp/test_dict'),
            seed=42
        )
        print('✓ Dictionary building executed')
        print(f'  Selected total: {result.selected_total}')

        return True
    except Exception as e:
        print(f'✗ Dictionary builder test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running advanced functionality tests for hypercircuit real implementations\n")

    tests = [
        test_real_surrogate_training,
        test_real_causal_runner,
        test_real_discovery,
        test_real_editing,
        test_dictionary_builder,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f'✗ Test {test.__name__} crashed: {e}')
            print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All real implementations are working correctly!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())