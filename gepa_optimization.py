# # -*- coding: utf-8 -*-
# """
# Final GEPA Optimization for PAN Card Extraction using DSPy
# """

# from gepa.api import optimize as gepa_optimize
# import dspy
# import os
# import json
# from gemma_adapter import GemmaAdapter  # Import your existing adapter

# # Import functions from evaluate_test_all_pan.py
# from evaluate_test_all_pan import (
#     load_complete_dataset, 
#     metric_with_feedback, 
#     GemmaPANExtractor,
#     PANCardExtraction,
#     _match
# )


# class PANCardGEPAAdapter(dspy.Adapter):
#     """
#     Custom GEPA Adapter for PAN Card Extraction optimization.
#     This is DIFFERENT from the GemmaAdapter - it's for GEPA optimization.
#     """
    
#     def __init__(self, metric_fn, failure_score=0.0):
#         super().__init__()
#         self.metric_fn = metric_fn
#         self.failure_score = failure_score
#         self.component_name = "extraction_instructions"
        
#     def candidate_to_program(self, candidate):
#         """Convert GEPA candidate to executable DSPy program."""
#         instructions = candidate.get(self.component_name, "")
        
#         # Create optimized signature with new instructions
#         class OptimizedPANCardExtraction(dspy.Signature):
#             __doc__ = instructions if instructions else PANCardExtraction.__doc__
#             image_url: str = dspy.InputField(desc="PAN card image URL")
#             name: str = dspy.OutputField(desc="Full name of cardholder")
#             date_of_birth: str = dspy.OutputField(desc="Date of birth in YYYY-MM-DD format")
#             id_number: str = dspy.OutputField(desc="PAN number (5 letters, 4 numbers, 1 letter)")
#             father_name: str = dspy.OutputField(desc="Father's name of cardholder")
#             has_income_tax_logo: str = dspy.OutputField(desc="TRUE if Income Tax Department logo visible")
#             has_gov_logo: str = dspy.OutputField(desc="TRUE if Govt. of India logo visible")
#             has_photo: str = dspy.OutputField(desc="TRUE if cardholder photo visible")
#             has_hologram_qrcode: str = dspy.OutputField(desc="TRUE if hologram or QR code visible")
#             has_national_emblem: str = dspy.OutputField(desc="TRUE if national emblem visible")
#             has_signature: str = dspy.OutputField(desc="TRUE if signature visible")
#             is_valid_pan: str = dspy.OutputField(desc="1 if PAN is valid, 0 if invalid")
        
#         class OptimizedExtractor(dspy.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.extract = dspy.Predict(OptimizedPANCardExtraction)
            
#             def forward(self, image_url: str):
#                 return self.extract(image_url=image_url)
        
#         return OptimizedExtractor()
    
#     def evaluate(self, candidate, devset, capture_traces=False):
#         """Evaluate candidate on development set."""
#         try:
#             program = self.candidate_to_program(candidate)
#             total_score = 0.0
#             successful_evals = 0
            
#             for example in devset:
#                 try:
#                     prediction = program(example.image_url)
#                     score = self.metric_fn(example, prediction)
#                     total_score += score
#                     successful_evals += 1
#                 except Exception as e:
#                     total_score += self.failure_score
            
#             avg_score = total_score / len(devset) if devset else self.failure_score
#             return avg_score
            
#         except Exception as e:
#             return self.failure_score


# def load_dataset_with_split(dev_fraction=0.22):
#     """
#     Load dataset and split into train/dev sets.
#     This replaces the missing function from evaluate_test_all_pan.py
#     """
#     all_examples = load_complete_dataset()
#     if not all_examples:
#         return [], []
    
#     split_idx = int(len(all_examples) * (1 - dev_fraction))
#     trainset = all_examples[:split_idx]
#     devset = all_examples[split_idx:]
    
#     return trainset, devset


# def create_reflection_lm():
#     """Create reflection LM using DSPy's configured Gemma model."""
#     def reflection_lm_function(prompt: str) -> str:
#         """Reflection LM using the same Gemma model."""
#         try:
#             # Use DSPy's configured LM for reflection
#             response = dspy.settings.lm(prompt)
#             return response
#         except Exception as e:
#             # Fallback reflection
#             return "Focus on accurately extracting all PAN card fields: name, PAN number, date of birth, father's name, and security features like logos, photos, holograms, and signatures."
    
#     return reflection_lm_function


# def ensure_directory_exists(directory_path):
#     """Ensure that a directory exists, create it if necessary."""
#     if not os.path.exists(directory_path):
#         os.makedirs(directory_path)
#         print(f"📁 Created directory: {directory_path}")
#     return directory_path


# def run_gepa_optimization(
#     max_metric_calls: int = 10,
#     dev_fraction: float = 0.22,
#     run_dir: str = "pan_card_gepa_runs",
#     api_key: str = None,
# ):
#     """
#     Run GEPA optimization for PAN card extraction.

#     Args:
#         max_metric_calls: Maximum metric evaluations
#         dev_fraction: Fraction for dev set
#         run_dir: Log directory
#         api_key: API key for Gemma

#     Returns:
#         Optimized candidate dict
#     """

#     api_key = api_key or "AIzaSyAuAsuPM3l0zA40tWfonxJdoyEOdrUWbk0"

#     # Configure DSPy with Gemma (same as evaluate_test_all_pan.py)
#     lm = dspy.LM(
#         model="gemini/gemma-3-27b-it",
#         api_key=api_key,
#         cache=False,
#     )
#     dspy.settings.configure(lm=lm, adapter=GemmaAdapter())  # Use your GemmaAdapter here

#     print("=" * 70)
#     print("GEPA Optimization for PAN Card Extraction")
#     print("=" * 70)

#     # -------------------------------------------------------------------------
#     # 0. Create necessary directories
#     # -------------------------------------------------------------------------
#     print(f"\n📋 Step 0: Creating necessary directories")
#     ensure_directory_exists(run_dir)
#     print("✅ Directories ready")

#     # -------------------------------------------------------------------------
#     # 1. Load Dataset
#     # -------------------------------------------------------------------------
#     print(f"\n📋 Step 1: Loading Dataset (dev_fraction={dev_fraction})")

#     trainset, devset = load_dataset_with_split(dev_fraction=dev_fraction)
#     print(f"✅ Dataset loaded: {len(trainset)} train, {len(devset)} dev")

#     # -------------------------------------------------------------------------
#     # 2. Create Custom GEPA Adapter
#     # -------------------------------------------------------------------------
#     print("\n📋 Step 2: Creating PAN Card GEPA Adapter")

#     adapter = PANCardGEPAAdapter(
#         metric_fn=metric_with_feedback,
#         failure_score=0.0,
#     )
#     print("✅ Custom adapter created")
#     print(f"   Component to optimize: {adapter.component_name}")

#     # -------------------------------------------------------------------------
#     # 3. Create Initial Candidate
#     # -------------------------------------------------------------------------
#     print("\n📋 Step 3: Creating Initial Candidate")

#     # Get baseline instructions from the current signature
#     baseline_instructions = """
#     Extract comprehensive information from PAN card images and validate authenticity.
#     Carefully read: cardholder name, date of birth, PAN number, father's name.
#     Check security features: Income Tax Department logo, Government of India logo, 
#     photograph, hologram/QR code, national emblem, and signature.
#     Determine validity based on presence of required security features.
#     Return all fields in the specified format.
#     """

#     initial_candidate = {adapter.component_name: baseline_instructions}
#     print("✅ Initial candidate created")
#     print(f"   Instructions preview: {baseline_instructions[:100]}...")

#     # -------------------------------------------------------------------------
#     # 4. Configure Reflection LM
#     # -------------------------------------------------------------------------
#     print("\n📋 Step 4: Configuring Reflection LM")

#     reflection_lm = create_reflection_lm()
#     print("✅ Using Gemma for reflection")

#     # -------------------------------------------------------------------------
#     # 5. Run GEPA Optimization
#     # -------------------------------------------------------------------------
#     print("\n" + "=" * 70)
#     print("🚀 Starting GEPA Optimization...")
#     print("=" * 70)
#     print(f"\nConfiguration:")
#     print(f"  - Max metric calls: {max_metric_calls}")
#     print(f"  - Train examples: {len(trainset)}")
#     print(f"  - Dev examples: {len(devset)}")
#     print(f"  - Log directory: {run_dir}")
#     print(f"\nOptimization progress:\n")

#     try:
#         result = gepa_optimize(
#             seed_candidate=initial_candidate,
#             adapter=adapter,
#             trainset=trainset,
#             valset=devset,
#             max_metric_calls=max_metric_calls,
#             reflection_lm=reflection_lm,
#             use_merge=True,
#             run_dir=run_dir,
#             display_progress_bar=True,
#         )

#         print("\n" + "=" * 70)
#         print("✅ GEPA Optimization Complete!")
#         print("=" * 70)

#         return result.best_candidate, initial_candidate

#     except Exception as e:
#         print(f"\n❌ GEPA optimization failed: {type(e).__name__}: {e}")
#         import traceback
#         traceback.print_exc()
#         raise


# def display_results(initial_candidate: dict, optimized_candidate: dict):
#     """Display optimization results."""
#     print("\n" + "=" * 70)
#     print("📊 Optimization Results")
#     print("=" * 70)

#     comp_name = "extraction_instructions"

#     print("\n🎯 Optimized Instructions:")
#     print("-" * 70)
#     optimized_instructions = optimized_candidate.get(comp_name, "N/A")
#     print(optimized_instructions)
#     print("-" * 70)

#     print("\n📋 Baseline Instructions (for comparison):")
#     print("-" * 70)
#     baseline_instructions = initial_candidate.get(comp_name, "N/A")
#     print(baseline_instructions)
#     print("-" * 70)

#     # Show statistics
#     initial_len = len(baseline_instructions)
#     optimized_len = len(optimized_instructions)
#     print(f"\n📈 Optimization Statistics:")
#     print(f"  - Baseline length: {initial_len} characters")
#     print(f"  - Optimized length: {optimized_len} characters")
#     print(f"  - Length change: {optimized_len - initial_len:+d} characters")


# def save_results(optimized_candidate: dict, filename: str = "pan_card_optimized.json"):
#     """Save optimized instructions."""
#     # Ensure the directory exists if filename includes path
#     directory = os.path.dirname(filename)
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
#         print(f"📁 Created directory for output: {directory}")

#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(optimized_candidate, f, indent=2, ensure_ascii=False)

#     print(f"\n💾 Optimized instructions saved to: {filename}")

#     # Also save as text file
#     text_filename = filename.replace(".json", ".txt")
#     with open(text_filename, "w", encoding="utf-8") as f:
#         f.write(optimized_candidate.get("extraction_instructions", ""))
    
#     print(f"💾 Instructions text saved to: {text_filename}")


# # =============================================================================
# # Main Execution
# # =============================================================================

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="GEPA optimization for PAN card extraction"
#     )
#     parser.add_argument("--max-metric-calls", type=int, default=10)
#     parser.add_argument("--dev-fraction", type=float, default=0.22)
#     parser.add_argument("--run-dir", default="pan_card_gepa_runs")
#     parser.add_argument("--save-output", default="pan_card_optimized.json")

#     args = parser.parse_args()

#     # Run optimization
#     optimized_candidate, initial_candidate = run_gepa_optimization(
#         max_metric_calls=args.max_metric_calls,
#         dev_fraction=args.dev_fraction,
#         run_dir=args.run_dir,
#     )

#     # Display results
#     display_results(initial_candidate, optimized_candidate)

#     # Save output
#     save_results(optimized_candidate, args.save_output)

#     print("\n" + "=" * 70)
#     print("🎉 GEPA Optimization Complete!")
#     print("=" * 70)
#     print(f"\n📁 Logs saved to: {args.run_dir}/")
#     print(f"💾 Optimized instructions saved to: {args.save_output}")
#     print("\nKey Features:")
#     print("  ✅ Uses your exact evaluation metric and dataset")
#     print("  ✅ Optimizes PAN card extraction instructions")
#     print("  ✅ Maintains compatibility with your existing code")

# -*- coding: utf-8 -*-



"""
GEPA Optimization for PAN Card Extraction using New DSPy GEPA Syntax
"""

import dspy
from dspy import GEPA
from dspy.evaluate import Evaluate
from gemma_adapter import GemmaAdapter
from evaluate_test_all_pan import (
    load_complete_dataset, 
    metric_with_feedback, 
    GemmaPANExtractor,
    PANCardExtraction
)

# Configure DSPy with Gemma
lmset = dspy.LM(
    model="openai/gpt-5-mini",  
    api_key="sk-b01398251e67d59f8322ffca84db8779365920c93efa7e72035b5a26fb2cfa54",  # OpenRouter key
    api_base="https://openrouter.ai/api/v1",  
    
    temperature=1.0,
    max_tokens=16000,
    cache=False,
)
dspy.settings.configure(lm=lmset, adapter=GemmaAdapter())

def run_gepa_optimization_new():
    """Run GEPA optimization using the new DSPy GEPA syntax"""
    
    print("=" * 70)
    print("🚀 GEPA Optimization for PAN Card Extraction (New Syntax)")
    print("=" * 70)
    
    # 1. Load dataset
    print("\n📋 Step 1: Loading Dataset")
    all_examples = load_complete_dataset()
    
    if not all_examples:
        raise ValueError("❌ No data loaded from dataset")
    
    # Split dataset (80% train, 20% dev)
    split_idx = int(len(all_examples) * 0.8)
    trainset, devset = all_examples[:split_idx], all_examples[split_idx:]
    
    print(f"✅ Dataset loaded: {len(trainset)} train, {len(devset)} dev examples")
    
    # 2. Define the base program
    print("\n📋 Step 2: Defining Base Program")
    base_program = GemmaPANExtractor()
    print("✅ Base program created")
    
    # 3. Define GEPA-friendly metric with feedback
    def gepa_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
        score = metric_with_feedback(example, prediction)
        feedback = f"Score: {score:.3f}" if score > 0.7 else f"Low score: {score:.3f}, needs improvement"
        return dspy.Prediction(score=score, feedback=feedback)
    
    print("✅ GEPA metric configured")
    
    # 4. Optimize with GEPA
    print("\n📋 Step 3: Configuring GEPA Optimization")
    
    gepa_optimizer = GEPA(
        metric=gepa_metric_with_feedback,
        reflection_lm=lmset,  # Use the same Gemma model for reflection
        #auto="light",      # Light optimization mode
        max_metric_calls=32     
    )
    
    print("✅ GEPA optimizer configured")
    
    # 5. Run GEPA compilation
    print("\n" + "=" * 70)
    print("🚀 Starting GEPA Optimization...")
    print("=" * 70)
    print(f"Training on {len(trainset)} examples, validating on {len(devset)} examples")
    print("⏳ This may take a while...\n")
    
    try:
        optimized_program = gepa_optimizer.compile(
            student=base_program,
            trainset=trainset,
            valset=devset
        )
        
        print("\n" + "=" * 70)
        print("✅ GEPA Optimization Complete!")
        print("=" * 70)
        
        return optimized_program, base_program, devset
        
    except Exception as e:
        print(f"\n❌ GEPA optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def evaluate_optimized_program(optimized_program, base_program, devset):
    """Evaluate both base and optimized programs"""
    print("\n" + "=" * 70)
    print("📊 Evaluating Programs")
    print("=" * 70)
    
    # Evaluate base program
    print("\n🔍 Evaluating Base Program...")
    base_evaluator = Evaluate(devset=devset, metric=metric_with_feedback, display_progress=True, display_table=3)
    base_result = base_evaluator(base_program)
    
    # Evaluate optimized program  
    print("\n🔍 Evaluating Optimized Program...")
    optimized_evaluator = Evaluate(devset=devset, metric=metric_with_feedback, display_progress=True, display_table=3)
    optimized_result = optimized_evaluator(optimized_program)
    
    # Display comparison
    print("\n" + "=" * 70)
    print("📈 PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Base Program Score:     {base_result.score:.3f}")
    print(f"Optimized Program Score: {optimized_result.score:.3f}")
    print(f"Improvement:            {optimized_result.score - base_result.score:+.3f}")
    
    return base_result.score, optimized_result.score

def main():
    """Main execution function"""
    try:
        # Run GEPA optimization
        optimized_program, base_program, devset = run_gepa_optimization_new()
        
        # Evaluate both programs
        base_score, optimized_score = evaluate_optimized_program(optimized_program, base_program, devset)
        optimized_prompt = optimized_program.extract.signature.__doc__

        # Final results
        print("\n" + "=" * 70)
        print("🎉 GEPA OPTIMIZATION COMPLETE!")
        print("=" * 70)
        print(f"📊 Final Results:")
        print(f"   - Base Model:      {base_score:.3f}")
        print(f"   - Optimized Model: {optimized_score:.3f}")
        print(f"   - Improvement:     {optimized_score - base_score:+.3f}")
        
        if optimized_score > base_score:
            print("   ✅ Optimization successful!")
        else:
            print("   ⚠️  No improvement achieved")

        print("🎯 FINAL OPTIMIZED PROMPT FROM GEPA:")
        print("=" * 70)
        print(optimized_prompt)
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

