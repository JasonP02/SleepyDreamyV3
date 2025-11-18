# Refactoring Plan: Action Handling in the World Model

This document outlines the plan to refactor the action handling mechanism within the SleepyDreamyV3 model.

## 1. Problem Diagnosis

The current implementation is suspected of passing action logits directly to the GRU in the world model's transition model. This is incorrect, as the recurrent state transition model (`next_state = f(current_state, action)`) requires a specific, sampled action to predict the next state accurately. The logits represent a probability distribution over actions, not the single action that was taken.

## 2. Proposed Refactoring Flow

The goal is to modify the data flow to ensure a sampled, one-hot encoded action is passed to the GRU.

### 2.1. Investigation Phase

1.  **Analyze `src/world_model.py`**:
    *   Locate the GRU cell within the `RSSM` (or equivalent) module.
    *   Trace the `action` tensor that is passed as input to the GRU cell during the forward pass.
    *   Confirm its origin and shape to determine if it's raw logits or a sampled action.

2.  **Analyze `src/trainer.py`**:
    *   Examine the main training loop.
    *   Identify where the action is generated (likely from an actor/policy network).
    *   Trace how this action is passed to the `world_model` for the dynamics prediction step.

### 2.2. Implementation Phase

1.  **Isolate Action Generation**:
    *   In `src/trainer.py`, after the actor network produces the action logits, we will introduce a step to sample from the resulting distribution.
    *   We can use a categorical distribution (e.g., `torch.distributions.categorical.Categorical`) to both sample the action index and calculate the log-probability for the policy loss.

2.  **Prepare Action for GRU**:
    *   The sampled action index will be converted into a one-hot encoded vector. `torch.nn.functional.one_hot` is the standard tool for this.
    *   This creates a vector of the correct shape (`[batch_size, num_actions]`) for the model.

3.  **Update World Model Input**:
    *   The newly created one-hot action vector will be passed to the world model's forward/transition function.
    *   This ensures the GRU receives the concrete action that was "taken".

4.  **Verify Downstream Usage**:
    *   Ensure that the original logits are still used where needed (e.g., for calculating the policy gradient loss). The sampled action is for the transition model, while the logits are for the policy update.

## 3. Next Steps

*   Begin the investigation phase by reading `src/world_model.py` and `src/trainer.py`.

Let's start by examining the code.
