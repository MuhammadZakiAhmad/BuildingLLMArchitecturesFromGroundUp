# Skeleton implementation of RLHF using GRPO
import re 
import torch 

def extract_final_number(text: str) -> str | None: 
    """Return the last integer or decimal number found in a completion.""" 
    matches = re.findall(r"-?\d+(?:\.\d+)?", text) 
    return matches[-1] if matches else None 

def verify_math_answer(completion: str, gold: str) -> float: 
    """A toy RLVR reward for answer-only math tasks.""" 
    predicted = extract_final_number(completion) 
    if predicted is None: 
        return 0.0 
    return 1.0 if predicted == str(gold) else 0.0



def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor: 
    """Normalize rewards within each prompt group.""" 
    mean = rewards.mean(dim=1, keepdim=True) 
    std = rewards.std(dim=1, keepdim=True).clamp_min(eps) 
    return (rewards - mean) / std

def grpo_loss( 
    logp: torch.Tensor, 
    old_logp: torch.Tensor, 
    ref_logp: torch.Tensor, 
    advantages: torch.Tensor, 
    eps: float = 0.2, 
    beta: float = 0.04, 
) -> torch.Tensor: 
    """Compute the clipped GRPO objective as a minimization loss.""" 
    ratio = torch.exp(logp - old_logp) 
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps) 
    adv = advantages.unsqueeze(-1) 
    
    surrogate = torch.minimum(ratio * adv, clipped * adv) 
    kl = logp - ref_logp 
    
    return -(surrogate - beta * kl).mean()

def train_step(batch, policy, reference, optimizer, group_size: int = 4): 
    """One GRPO + RLVR training step skeleton.""" 
    prompts, gold_answers = batch 
    
    # 1. Generate a group of completions for the prompts
    completions, old_logp = sample_group(policy, prompts, group_size) 
  
    # 2. Score the completions using our verifier
    rewards = torch.tensor( 
        [ 
            [verify_math_answer(y, gold) for y in group] 
            for group, gold in zip(completions, gold_answers) 
        ], 
        device=old_logp.device, 
    ) 
    
    # 3. Calculate advantages
    advantages = group_advantages(rewards) 
  
    # 4. Get token probabilities from current and reference models
    logp = sequence_logprobs(policy, prompts, completions) 
    with torch.no_grad(): 
        ref_logp = sequence_logprobs(reference, prompts, completions) 
  
    # 5. Compute loss and optimize
    loss = grpo_loss(logp, old_logp, ref_logp, advantages) 
    optimizer.zero_grad(set_to_none=True) 
    loss.backward() 
    optimizer.step() 
    
    return {"loss": float(loss.detach()), "reward": float(rewards.mean())}