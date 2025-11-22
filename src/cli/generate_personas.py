"""CLI for generating personas."""

import argparse
from pathlib import Path
import pickle

from src.models.persona_generator.llm_persona_engine import PersonaGenerator


def main():
    """Main persona generation function."""
    parser = argparse.ArgumentParser(description='Generate neurodivergent personas')
    parser.add_argument('--num-asd', type=int, default=400,
                       help='Number of ASD personas')
    parser.add_argument('--num-adhd', type=int, default=400,
                       help='Number of ADHD personas')
    parser.add_argument('--num-nt', type=int, default=200,
                       help='Number of NT personas')
    parser.add_argument('--output-dir', type=str, default='data/synthetic/personas',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='LLM model name')
    
    args = parser.parse_args()
    
    print(f"Generating personas...")
    print(f"ASD: {args.num_asd}, ADHD: {args.num_adhd}, NT: {args.num_nt}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = PersonaGenerator(model_name=args.model)
    
    # Generate personas
    print("Generating ASD personas...")
    asd_personas = generator.batch_generate(args.num_asd, 'ASD')
    
    print("Generating ADHD personas...")
    adhd_personas = generator.batch_generate(args.num_adhd, 'ADHD')
    
    print("Generating NT personas...")
    nt_personas = generator.batch_generate(args.num_nt, 'NT')
    
    # Save personas
    with open(output_dir / 'asd_personas.pkl', 'wb') as f:
        pickle.dump(asd_personas, f)
    
    with open(output_dir / 'adhd_personas.pkl', 'wb') as f:
        pickle.dump(adhd_personas, f)
    
    with open(output_dir / 'nt_personas.pkl', 'wb') as f:
        pickle.dump(nt_personas, f)
    
    print(f"Personas saved to {output_dir}")
    print(f"Total: {len(asd_personas) + len(adhd_personas) + len(nt_personas)} personas")


if __name__ == '__main__':
    main()
