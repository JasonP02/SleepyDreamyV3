import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from lunar_lander import create_lunar_lander_with_vision
import time
import cv2

def resize_frame(frame, target_size=(150, 100)):
    """
    Resize frame to target size without cropping using bilinear interpolation.
    
    Args:
        frame: Input frame as numpy array
        target_size: Target size as (width, height)
    
    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def visualize_simulation_frames(num_episodes=1, max_steps_per_episode=500, delay=0.05):
    """
    Visualizes simulation frames from the Lunar Lander environment.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        delay: Delay between frames in seconds (for visualization)
    """
    env = create_lunar_lander_with_vision()
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        # Set up the plot
        plt.figure(figsize=(6, 4))
        
        for step in range(max_steps_per_episode):
            # Clear the previous frame
            plt.clf()
            
            # Resize the frame
            import torch.nn.functional as F
            import torch
            pixel_tensor = torch.tensor(obs['pixels']).permute(2,0,1).unsqueeze(0)
            resized_frame = F.interpolate(pixel_tensor, size=(64,64), mode='bilinear', align_corners=False)
            resized_frame = resized_frame.squeeze(0).permute(1,2,0)
            
            # Display the current frame
            plt.imshow(resized_frame)
            plt.title(f"Episode {episode + 1}, Step {step + 1}, Reward: {episode_reward:.2f}")
            plt.axis('off')
            
            # Add state information as text
            state_text = f"State: {obs['state']}"
            plt.text(5, resized_frame.shape[0] - 5, state_text,
                    color='white', fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
            
            plt.pause(delay)  # Pause to create animation effect
            
            # Take a random action (you could replace this with your policy)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"  Episode finished after {step_count + 1} steps with total reward: {episode_reward:.2f}")
                break
        
        plt.close()
    
    env.close()

def save_frames_to_video(num_episodes=1, max_steps_per_episode=500, output_path="lunar_lander_frames.mp4"):
    """
    Saves simulation frames to a video file.
    
    Args:
        num_episodes: Number of episodes to record
        max_steps_per_episode: Maximum steps per episode
        output_path: Path to save the video
    """
    env = create_lunar_lander_with_vision()
    
    # Get frame dimensions from the first observation
    obs, info = env.reset()
    
    # Resize frame to target size
    resized_frame = resize_frame(obs['pixels'], target_size=(150, 100))
    frame_height, frame_width = resized_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    
    for episode in range(num_episodes):
        print(f"Recording Episode {episode + 1}")
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        for step in range(max_steps_per_episode):
            # Resize frame
            resized_frame = resize_frame(obs['pixels'], target_size=(150, 100))
            
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
            
            # Add episode and step information to the frame (adjusted for smaller size)
            cv2.putText(frame, f"Ep {episode + 1}, St {step + 1}",
                       (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"R: {episode_reward:.2f}",
                       (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Write frame to video
            out.write(frame)
            
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"  Episode finished after {step_count + 1} steps with total reward: {episode_reward:.2f}")
                break
    
    # Release video writer
    out.release()
    env.close()
    print(f"Video saved to {output_path}")

def analyze_observation_structure():
    """
    Analyzes and displays the structure of observations from the environment.
    """
    env = create_lunar_lander_with_vision()
    
    print("Environment Information:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset and analyze first observation
    obs, info = env.reset()
    
    print("\nFirst Observation Structure:")
    print(f"Keys: {obs.keys()}")
    print(f"State shape: {obs['state'].shape}")
    print(f"State dtype: {obs['state'].dtype}")
    print(f"State range: [{np.min(obs['state']):.3f}, {np.max(obs['state']):.3f}]")
    
    print(f"Original pixels shape: {obs['pixels'].shape}")
    print(f"Pixels dtype: {obs['pixels'].dtype}")
    print(f"Pixels range: [{np.min(obs['pixels'])}, {np.max(obs['pixels'])}]")
    
    # Show resized frame
    resized_frame = resize_frame(obs['pixels'], target_size=(150, 100))
    print(f"Resized pixels shape: {resized_frame.shape}")
    print(f"Resized pixels dtype: {resized_frame.dtype}")
    print(f"Resized pixels range: [{np.min(resized_frame)}, {np.max(resized_frame)}]")
    
    # Take a few steps and show how observations change
    print("\nObservation changes over 5 steps:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        resized_frame = resize_frame(obs['pixels'], target_size=(150, 100))
        
        print(f"Step {i+1}:")
        print(f"  State: {obs['state']}")
        print(f"  State mean: {np.mean(obs['state']):.3f}, std: {np.std(obs['state']):.3f}")
        print(f"  Original pixels mean: {np.mean(obs['pixels']):.3f}, std: {np.std(obs['pixels']):.3f}")
        print(f"  Resized pixels mean: {np.mean(resized_frame):.3f}, std: {np.std(resized_frame):.3f}")
        print(f"  Reward: {reward:.3f}")
        
        if terminated or truncated:
            break
    
    env.close()

if __name__ == "__main__":
    print("Choose what to do:")
    print("1. Visualize simulation frames (requires matplotlib)")
    print("2. Save frames to video (requires opencv-python)")
    print("3. Analyze observation structure")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        try:
            import matplotlib.pyplot as plt
            num_episodes = int(input("Number of episodes to visualize (default 1): ") or "1")
            delay = float(input("Delay between frames in seconds (default 0.05): ") or "0.05")
            visualize_simulation_frames(num_episodes=num_episodes, delay=delay)
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")
    
    elif choice == "2":
        num_episodes = int(input("Number of episodes to record (default 1): ") or "1")
        output_path = input("Output video path (default 'lunar_lander_frames.mp4'): ") or "lunar_lander_frames.mp4"
        save_frames_to_video(num_episodes=num_episodes, output_path=output_path)
    
    elif choice == "3":
        analyze_observation_structure()
    
    else:
        print("Invalid choice. Running observation structure analysis...")
        analyze_observation_structure()