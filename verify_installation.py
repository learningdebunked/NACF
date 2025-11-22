#!/usr/bin/env python3
"""Verify NACF installation and components."""

import sys
from pathlib import Path


def check_imports():
    """Check if all core modules can be imported."""
    print("=" * 60)
    print("Checking Core Imports...")
    print("=" * 60)
    
    imports = [
        ("TAN Model", "src.models.tan.temporal_attention_network", "TAN"),
        ("CNN Layer", "src.models.tan.cnn_layer", "CNN1DLayer"),
        ("GRU Layer", "src.models.tan.gru_layer", "GRULayer"),
        ("Attention", "src.models.tan.attention_layer", "MultiHeadAttention"),
        ("Persona Generator", "src.models.persona_generator.llm_persona_engine", "PersonaGenerator"),
        ("RL Environment", "src.models.rl.environment", "CheckoutEnvironment"),
        ("A2C Agent", "src.models.rl.a2c_agent", "A2CAgent"),
        ("Policy Network", "src.models.rl.policy_network", "PolicyNetwork"),
        ("Value Network", "src.models.rl.value_network", "ValueNetwork"),
        ("Dataset", "src.data.datasets.clickstream_dataset", "ClickstreamDataset"),
        ("Sequence Builder", "src.data.preprocessing.sequence_builder", "SequenceBuilder"),
        ("Feature Extractor", "src.data.preprocessing.feature_engineering", "FeatureExtractor"),
        ("Metrics", "src.evaluation.metrics", "calculate_auc"),
        ("TAN Trainer", "src.training.tan_trainer", "TANTrainer"),
    ]
    
    success = 0
    failed = []
    
    for name, module, cls in imports:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"✓ {name:30s} OK")
            success += 1
        except Exception as e:
            print(f"✗ {name:30s} FAILED: {e}")
            failed.append(name)
    
    print(f"\n{success}/{len(imports)} imports successful")
    
    if failed:
        print(f"\nFailed imports: {', '.join(failed)}")
        return False
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "=" * 60)
    print("Checking Dependencies...")
    print("=" * 60)
    
    deps = [
        "torch",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "transformers",
        "gymnasium",
    ]
    
    success = 0
    failed = []
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep:20s} OK")
            success += 1
        except ImportError:
            print(f"✗ {dep:20s} NOT FOUND")
            failed.append(dep)
    
    print(f"\n{success}/{len(deps)} dependencies found")
    
    if failed:
        print(f"\nMissing dependencies: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True


def check_structure():
    """Check if directory structure is correct."""
    print("\n" + "=" * 60)
    print("Checking Directory Structure...")
    print("=" * 60)
    
    required_dirs = [
        "src/config",
        "src/data/loaders",
        "src/data/preprocessing",
        "src/data/datasets",
        "src/models/tan",
        "src/models/persona_generator",
        "src/models/rl",
        "src/training",
        "src/evaluation",
        "src/cli",
        "experiments/hypothesis_testing",
        "tests/unit",
        "tests/integration",
        "data/raw",
        "data/processed",
        "data/synthetic",
        "results",
    ]
    
    success = 0
    failed = []
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path:40s} OK")
            success += 1
        else:
            print(f"✗ {dir_path:40s} NOT FOUND")
            failed.append(dir_path)
    
    print(f"\n{success}/{len(required_dirs)} directories found")
    
    if failed:
        print(f"\nMissing directories: {', '.join(failed)}")
        return False
    return True


def check_configs():
    """Check if configuration files exist."""
    print("\n" + "=" * 60)
    print("Checking Configuration Files...")
    print("=" * 60)
    
    config_files = [
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "README.md",
        "LICENSE",
    ]
    
    success = 0
    
    for file_path in config_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path:30s} OK")
            success += 1
        else:
            print(f"✗ {file_path:30s} NOT FOUND")
    
    print(f"\n{success}/{len(config_files)} config files found")
    return success == len(config_files)


def run_quick_test():
    """Run a quick functional test."""
    print("\n" + "=" * 60)
    print("Running Quick Functional Test...")
    print("=" * 60)
    
    try:
        import torch
        import numpy as np
        from src.models.tan.temporal_attention_network import TAN
        
        print("Creating TAN model...")
        model = TAN(input_dim=3, cnn_filters=64, gru_hidden=128, attention_heads=4)
        
        print("Creating test input...")
        x = torch.randn(4, 50, 3)
        
        print("Running forward pass...")
        output = model(x)
        
        assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"
        assert torch.all((output >= 0) & (output <= 1)), "Output not in [0, 1] range"
        
        print("✓ TAN model test passed")
        
        # Test persona generator
        from src.models.persona_generator.llm_persona_engine import PersonaGenerator
        
        print("\nTesting persona generator...")
        generator = PersonaGenerator()
        personas = generator.batch_generate(5, 'ASD')
        
        assert len(personas) == 5, f"Expected 5 personas, got {len(personas)}"
        print(f"✓ Generated {len(personas)} personas")
        
        # Test RL environment
        from src.models.rl.environment import CheckoutEnvironment
        
        print("\nTesting RL environment...")
        env = CheckoutEnvironment(personas)
        state, _ = env.reset()
        
        assert len(state) == 64, f"Expected state dim 64, got {len(state)}"
        print(f"✓ RL environment initialized")
        
        print("\n" + "=" * 60)
        print("✓ All functional tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Functional test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("NACF Installation Verification")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_structure),
        ("Configuration Files", check_configs),
        ("Dependencies", check_dependencies),
        ("Core Imports", check_imports),
        ("Functional Tests", run_quick_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("✓ All checks passed! NACF is ready to use.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Generate personas: nacf-generate-personas --num-asd 100")
        print("2. Train model: nacf-train --model tan --epochs 10")
        print("3. Run experiment: python experiments/hypothesis_testing/h1_friction_detection/train_tan.py")
        print("4. See QUICKSTART.md for more examples")
        return 0
    else:
        print("\n" + "=" * 60)
        print("✗ Some checks failed. Please review errors above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
