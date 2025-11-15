# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the SleepyDreamyV3 project.

## Project Overview

This project is a PyTorch implementation of the DreamerV3 paper, a model-based reinforcement learning agent. The current implementation supports the `LunarLander-v3` environment from Gymnasium.

The core of the project is the `RSSMWorldModel` (Recurrent State Space Model), which is a world model that learns a compressed representation of the environment's state and dynamics. The agent uses this world model to learn a policy (actor) and a value function (critic).

The main components of the architecture are:

*   **`RSSMWorldModel`**: The core world model, which includes:
    *   **`ObservationEncoder`**: Encodes observations into a latent space.
    *   **`GatedRecurrentUnit` (GRU)**: The recurrent component of the model.
    *   **`DynamicsPredictor`**: Predicts the next latent state.
    *   **`ObservationDecoder`**: Reconstructs observations from the latent state.
*   **`ThreeLayerMLP`**: A simple MLP used for the actor and critic networks.
*   **Configuration**: The project uses `pydantic` for configuration, defined in `src/config.py`.

## Building and Running

The project uses `uv` for dependency management.

### Running the Application

To run the main application (training and environment interaction), use the following command:

```bash
uv run python -m src.main
```

### Running Tests

The project uses `pytest` for testing. To run the tests, use the following command:

```bash
uv run pytest
```

## Development Conventions

*   **Configuration**: All configuration is managed through `pydantic` models in `src/config.py`. This provides a single source of truth for all hyperparameters and settings.
*   **Testing**: Tests are located in the `tests/` directory and are run using `pytest`. The current tests cover component shapes and basic forward passes. The `test_plans.md` file outlines a plan for more comprehensive testing.
*   **Code Style**: The code follows standard Python conventions (PEP 8). The `.ruff_cache` directory suggests that `ruff` is used for linting and formatting.
