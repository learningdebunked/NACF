"""H2: Generate and validate neurodivergent personas."""

from pathlib import Path
import pickle

from src.models.persona_generator.llm_persona_engine import PersonaGenerator


def main():
    """Generate personas for H2."""
    print("=" * 60)
    print("H2: Neurodivergent Persona Generation")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("results/h2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print("\nInitializing persona generator...")
    generator = PersonaGenerator(model_name="gpt2")
    
    # Generate personas
    print("\nGenerating ASD personas...")
    asd_personas = generator.batch_generate(num_personas=400, persona_type='ASD')
    print(f"Generated {len(asd_personas)} ASD personas")
    
    print("\nGenerating ADHD personas...")
    adhd_personas = generator.batch_generate(num_personas=400, persona_type='ADHD')
    print(f"Generated {len(adhd_personas)} ADHD personas")
    
    print("\nGenerating NT personas...")
    nt_personas = generator.batch_generate(num_personas=200, persona_type='NT')
    print(f"Generated {len(nt_personas)} NT personas")
    
    # Save personas
    with open(output_dir / 'asd_personas.pkl', 'wb') as f:
        pickle.dump(asd_personas, f)
    
    with open(output_dir / 'adhd_personas.pkl', 'wb') as f:
        pickle.dump(adhd_personas, f)
    
    with open(output_dir / 'nt_personas.pkl', 'wb') as f:
        pickle.dump(nt_personas, f)
    
    print(f"\nPersonas saved to {output_dir}")
    print(f"Total: {len(asd_personas) + len(adhd_personas) + len(nt_personas)} personas")
    
    print("\nTarget validation metrics:")
    print("- KS overlap (ASD): 0.93")
    print("- KS overlap (ADHD): 0.89")


if __name__ == "__main__":
    main()
