import torch
from src.environment import create_env_with_vision
from src.trainer_utils import initialize_actor, initialize_world_model

def test_env_init():
    print("Testing environment initialization...")
    try:
        env = create_env_with_vision()
        print("Environment created successfully.")
        env.close()
    except Exception as e:
        print(f"Failed to create environment: {e}")
        raise

    print("Testing model initialization (as in collect_experiences)...")
    device = "cpu"
    try:
        actor = initialize_actor(device=device)
        print("Actor initialized successfully.")
        
        encoder, world_model = initialize_world_model(device, batch_size=1)
        print("Encoder and World Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        raise

    print("Done.")

if __name__ == "__main__":
    test_env_init()
