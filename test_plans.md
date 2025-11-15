# Test Plans

This document outlines potential testing strategies to improve the robustness and reliability of the project.

## 1. Pipeline & Integration Tests

These tests verify that different components of the system work together correctly.

*   **World Model Training Step Test:**
    *   **Goal:** Simulate a single, complete training step for the `RSSMWorldModel`.
    *   **Checks:**
        *   Data flows through the model without errors.
        *   Loss is calculated correctly.
        *   Gradients are computed and weights are updated (i.e., no detached tensors).

*   **Full Agent-Environment Interaction Test:**
    *   **Goal:** Run a short "episode" (5-10 steps) to test the full agent-environment loop.
    *   **Checks:**
        *   Agent selects an action.
        *   Environment executes the action.
        *   World model processes the new observation.
        *   Actor and critic are updated.

## 2. Model Component Tests

These tests focus on the behavior of individual model components.

*   **Gradient Flow Test:**
    *   **Goal:** Ensure all model parameters are learning.
    *   **Checks:**
        *   After a backward pass, check that `.grad` is not `None` or all zeros for all trainable parameters.

*   **Distribution Output Test:**
    *   **Goal:** Verify the probabilistic outputs of the `RSSMWorldModel`.
    *   **Checks:**
        *   `posterior_dist` and `prior_dist` are valid `torch.distributions.Categorical` objects.
        *   Check for correct event shapes.

## 3. Data and Environment Tests

These tests ensure the data pipeline and environment interactions are reliable.

*   **Data Collection Test:**
    *   **Goal:** Verify the `collect_bootstrapping_examples` function.
    *   **Checks:**
        *   The function runs without errors.
        *   The output file is created.
        *   Data within the file has the expected format and data types.
