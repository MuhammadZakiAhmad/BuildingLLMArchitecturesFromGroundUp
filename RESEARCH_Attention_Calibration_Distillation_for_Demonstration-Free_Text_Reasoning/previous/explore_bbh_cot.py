import random
from datasets import load_dataset
import os

def main():
    # We will sample from 5 reasoning-heavy BBH tasks (20 from each to make 100)
    tasks = [
        "boolean_expressions", 
        "navigate", 
        "snarks", 
        "logical_deduction_five_objects",
        "date_understanding"
    ]
    
    # lukaemon/bbh is the most reliable HuggingFace BBH repository
    dataset_name = "lukaemon/bbh"
    
    print(f"Connecting to Hugging Face to load {dataset_name}...")
    
    all_examples = []
    
    for task in tasks:
        print(f"Fetching task: {task}...")
        try:
            # Load the test split for the specific task
            dataset = load_dataset(dataset_name, task, split="test")
            
            # Randomly sample 20 examples from each task
            population_size = len(dataset)
            sample_size = min(20, population_size)
            
            sampled_indices = random.sample(range(population_size), sample_size)
            sampled_data = dataset.select(sampled_indices)
            
            for item in sampled_data:
                all_examples.append({
                    "task": task,
                    "input": item.get("input", "N/A"),
                    "target": item.get("target", "N/A")
                })
        except Exception as e:
            print(f"Error loading {task}: {e}")
            
    # Randomize the final list of 100 examples
    random.shuffle(all_examples)
    
    # Save to a text file for easy reading
    output_filename = "100_random_bbh_examples.txt"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"=== 100 Random Examples from {dataset_name} ===\n")
        f.write("Note: The Chain-of-Thought (CoT) prompts are standard 3-shot templates applied to these base inputs.\n\n")
        
        for i, ex in enumerate(all_examples):
            f.write(f"Example {i+1} | Task: {ex['task']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"INPUT (Student sees this 0-shot):\n{ex['input']}\n\n")
            f.write(f"TARGET (Teacher arrives at this via CoT):\n{ex['target']}\n")
            f.write("=" * 80 + "\n\n")
            
    print(f"\nSuccessfully downloaded and sampled {len(all_examples)} examples!")
    print(f"Saved to: {os.path.abspath(output_filename)}")
    print("Open this file in your editor to review the data structure.")

if __name__ == "__main__":
    main()
