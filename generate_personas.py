#!/usr/bin/env python3
"""Generate neurodivergent personas for NACF."""

import pickle
from pathlib import Path
import numpy as np

from src.models.persona_generator.llm_persona_engine import PersonaGenerator


def generate_all_personas(num_asd=100, num_adhd=100, num_nt=50):
    """Generate all persona types."""
    print("=" * 70)
    print("NACF Persona Generation")
    print("=" * 70)
    
    print(f"\nGenerating:")
    print(f"  • {num_asd} ASD personas")
    print(f"  • {num_adhd} ADHD personas")
    print(f"  • {num_nt} NT (neurotypical) personas")
    print(f"  Total: {num_asd + num_adhd + num_nt} personas")
    
    # Initialize generator
    print("\n[1/4] Initializing persona generator...")
    generator = PersonaGenerator(model_name="gpt2")
    print("✓ Generator initialized")
    
    # Generate ASD personas
    print(f"\n[2/4] Generating {num_asd} ASD personas...")
    asd_personas = generator.batch_generate(num_asd, 'ASD')
    print(f"✓ Generated {len(asd_personas)} ASD personas")
    
    # Show example ASD persona
    if len(asd_personas) > 0:
        example = asd_personas[0]
        print(f"\n  Example ASD Persona:")
        print(f"    ID: {example.persona_id}")
        print(f"    Type: {example.persona_type}")
        print(f"    Traits:")
        for trait, value in example.traits.items():
            print(f"      • {trait}: {value:.2f}")
        print(f"    Behavioral Parameters:")
        for param, value in list(example.behavioral_params.items())[:3]:
            print(f"      • {param}: {value:.2f}")
    
    # Generate ADHD personas
    print(f"\n[3/4] Generating {num_adhd} ADHD personas...")
    adhd_personas = generator.batch_generate(num_adhd, 'ADHD')
    print(f"✓ Generated {len(adhd_personas)} ADHD personas")
    
    # Show example ADHD persona
    if len(adhd_personas) > 0:
        example = adhd_personas[0]
        print(f"\n  Example ADHD Persona:")
        print(f"    ID: {example.persona_id}")
        print(f"    Type: {example.persona_type}")
        print(f"    Traits:")
        for trait, value in example.traits.items():
            print(f"      • {trait}: {value:.2f}")
    
    # Generate NT personas
    print(f"\n[4/4] Generating {num_nt} NT personas...")
    nt_personas = generator.batch_generate(num_nt, 'NT')
    print(f"✓ Generated {len(nt_personas)} NT personas")
    
    return asd_personas, adhd_personas, nt_personas


def analyze_personas(asd_personas, adhd_personas, nt_personas):
    """Analyze generated personas."""
    print("\n" + "=" * 70)
    print("Persona Analysis")
    print("=" * 70)
    
    # ASD analysis
    print("\n📊 ASD Personas:")
    if len(asd_personas) > 0:
        sensitivities = [p.traits.get('sensory_sensitivity', 0) for p in asd_personas]
        repetitive = [p.traits.get('repetitive_behavior', 0) for p in asd_personas]
        print(f"  Sensory Sensitivity: {np.mean(sensitivities):.2f} ± {np.std(sensitivities):.2f}")
        print(f"  Repetitive Behavior: {np.mean(repetitive):.2f} ± {np.std(repetitive):.2f}")
        print(f"  Avg Abandonment Threshold: {np.mean([p.behavioral_params['abandonment_threshold'] for p in asd_personas]):.2f}")
    
    # ADHD analysis
    print("\n📊 ADHD Personas:")
    if len(adhd_personas) > 0:
        attention = [p.traits.get('attention_variance', 0) for p in adhd_personas]
        impulsivity = [p.traits.get('impulsivity', 0) for p in adhd_personas]
        print(f"  Attention Variance: {np.mean(attention):.2f} ± {np.std(attention):.2f}")
        print(f"  Impulsivity: {np.mean(impulsivity):.2f} ± {np.std(impulsivity):.2f}")
        print(f"  Avg Click Rate: {np.mean([p.behavioral_params['click_rate_mean'] for p in adhd_personas]):.2f}")
    
    # NT analysis
    print("\n📊 NT Personas:")
    if len(nt_personas) > 0:
        print(f"  Balanced traits (moderate values)")
        print(f"  Avg Abandonment Threshold: {np.mean([p.behavioral_params['abandonment_threshold'] for p in nt_personas]):.2f}")
    
    # Compare groups
    print("\n📈 Group Comparison:")
    print(f"  ASD: Lower abandonment threshold (more sensitive)")
    print(f"  ADHD: Higher click rate variance (more impulsive)")
    print(f"  NT: Balanced behavior (baseline)")


def save_personas(asd_personas, adhd_personas, nt_personas, output_dir='data/synthetic/personas'):
    """Save personas to disk."""
    print("\n" + "=" * 70)
    print("Saving Personas")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each type
    with open(output_path / 'asd_personas.pkl', 'wb') as f:
        pickle.dump(asd_personas, f)
    print(f"✓ Saved {len(asd_personas)} ASD personas to {output_path / 'asd_personas.pkl'}")
    
    with open(output_path / 'adhd_personas.pkl', 'wb') as f:
        pickle.dump(adhd_personas, f)
    print(f"✓ Saved {len(adhd_personas)} ADHD personas to {output_path / 'adhd_personas.pkl'}")
    
    with open(output_path / 'nt_personas.pkl', 'wb') as f:
        pickle.dump(nt_personas, f)
    print(f"✓ Saved {len(nt_personas)} NT personas to {output_path / 'nt_personas.pkl'}")
    
    # Save combined
    all_personas = asd_personas + adhd_personas + nt_personas
    with open(output_path / 'all_personas.pkl', 'wb') as f:
        pickle.dump(all_personas, f)
    print(f"✓ Saved {len(all_personas)} total personas to {output_path / 'all_personas.pkl'}")
    
    # Save summary
    with open(output_path / 'persona_summary.txt', 'w') as f:
        f.write("NACF Persona Generation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Personas: {len(all_personas)}\n")
        f.write(f"  ASD: {len(asd_personas)}\n")
        f.write(f"  ADHD: {len(adhd_personas)}\n")
        f.write(f"  NT: {len(nt_personas)}\n\n")
        f.write("Files:\n")
        f.write(f"  - asd_personas.pkl\n")
        f.write(f"  - adhd_personas.pkl\n")
        f.write(f"  - nt_personas.pkl\n")
        f.write(f"  - all_personas.pkl\n")
    
    print(f"✓ Saved summary to {output_path / 'persona_summary.txt'}")
    
    return output_path


def demonstrate_usage(personas):
    """Show how to use the generated personas."""
    print("\n" + "=" * 70)
    print("Usage Example")
    print("=" * 70)
    
    print("\nTo load and use personas in your code:")
    print("""
import pickle

# Load personas
with open('data/synthetic/personas/all_personas.pkl', 'rb') as f:
    personas = pickle.load(f)

# Use in RL environment
from src.models.rl.environment import CheckoutEnvironment
env = CheckoutEnvironment(personas)

# Or filter by type
asd_personas = [p for p in personas if p.persona_type == 'ASD']
adhd_personas = [p for p in personas if p.persona_type == 'ADHD']
""")
    
    # Show a sample persona
    if len(personas) > 0:
        print("\nSample Persona Structure:")
        p = personas[0]
        print(f"  persona_id: '{p.persona_id}'")
        print(f"  persona_type: '{p.persona_type}'")
        print(f"  traits: {dict(list(p.traits.items())[:2])}...")
        print(f"  behavioral_params: {dict(list(p.behavioral_params.items())[:2])}...")


def main():
    """Main persona generation function."""
    print("\n" + "=" * 70)
    print("NACF Neurodivergent Persona Generator")
    print("=" * 70)
    
    # Generate personas
    asd_personas, adhd_personas, nt_personas = generate_all_personas(
        num_asd=100,
        num_adhd=100,
        num_nt=50
    )
    
    # Analyze
    analyze_personas(asd_personas, adhd_personas, nt_personas)
    
    # Save
    output_path = save_personas(asd_personas, adhd_personas, nt_personas)
    
    # Show usage
    all_personas = asd_personas + adhd_personas + nt_personas
    demonstrate_usage(all_personas)
    
    # Summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"\n✅ Successfully generated {len(all_personas)} personas!")
    print(f"   • {len(asd_personas)} ASD personas")
    print(f"   • {len(adhd_personas)} ADHD personas")
    print(f"   • {len(nt_personas)} NT personas")
    print(f"\n📁 Saved to: {output_path}")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review personas in {output_path}")
    print(f"   2. Train RL agent: python3 train_rl_agent.py")
    print(f"   3. Run experiments with personas")


if __name__ == "__main__":
    main()
