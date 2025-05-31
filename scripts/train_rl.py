"""
MIT License

Copyright (c) 2024 kunalsingh2514@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
RL Agent Training Script for TMS2
Specialized script for training reinforcement learning signal control agents.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.config_manager import init_config
from src.utils.logger import setup_logging, get_logger
from src.models.rl_agent import RLCoordinator, RLState
from src.training.training_utils import TrafficDataGenerator, ModelEvaluator


class TrafficEnvironment:
    """
    Simplified traffic environment for RL training.
    """
    
    def __init__(self, intersection_ids, data_generator=None):
        """Initialize the traffic environment."""
        self.intersection_ids = intersection_ids
        self.data_generator = data_generator or TrafficDataGenerator()
        self.logger = get_logger("TrafficEnvironment")
        
        # Environment state
        self.current_states = {}
        self.episode_step = 0
        self.max_episode_steps = 200
        
        # Traffic patterns
        self.traffic_patterns = {}
        self._generate_traffic_patterns()
        
        self.reset()
    
    def _generate_traffic_patterns(self):
        """Generate traffic patterns for each intersection."""
        for intersection_id in self.intersection_ids:
            # Generate daily traffic pattern
            daily_pattern = self.data_generator.generate_daily_traffic_pattern(24)
            
            # Extend to hourly resolution for simulation
            hourly_pattern = []
            for hour_traffic in daily_pattern:
                for minute in range(60):
                    variation = np.random.normal(0, 0.1)
                    minute_traffic = max(0, hour_traffic + variation)
                    hourly_pattern.append(minute_traffic)
            
            self.traffic_patterns[intersection_id] = hourly_pattern
    
    def reset(self):
        """Reset the environment for a new episode."""
        self.episode_step = 0
        
        # Initialize states for all intersections
        for intersection_id in self.intersection_ids:
            self.current_states[intersection_id] = self._generate_initial_state(intersection_id)
        
        return self.current_states
    
    def _generate_initial_state(self, intersection_id):
        """Generate initial state for an intersection."""
        # Get traffic level from pattern
        pattern_idx = self.episode_step % len(self.traffic_patterns[intersection_id])
        traffic_level = self.traffic_patterns[intersection_id][pattern_idx]
        
        base_vehicles = int(traffic_level * 30)  # 0-30 vehicles per lane
        vehicle_counts = [
            max(0, base_vehicles + np.random.randint(-5, 6)) for _ in range(4)
        ]
        
        return RLState(
            vehicle_counts=vehicle_counts,
            current_phase=np.random.randint(0, 4),
            phase_time=np.random.randint(0, 60),
            queue_lengths=[max(0, vc - 10) for vc in vehicle_counts],
            waiting_times=[np.random.uniform(0, 120) for _ in range(4)],
            lstm_predictions=[vc + np.random.normal(0, 2) for vc in vehicle_counts]
        )
    
    def step(self, actions):
        """
        Execute actions and return new states, rewards, and done flags.
        
        Args:
            actions: Dictionary mapping intersection_id to action
            
        Returns:
            Tuple of (new_states, rewards, done, info)
        """
        new_states = {}
        rewards = {}
        
        for intersection_id, action in actions.items():
            current_state = self.current_states[intersection_id]
            new_state = self._update_state(intersection_id, current_state, action)
            
            reward = self._calculate_reward(current_state, action, new_state)
            
            new_states[intersection_id] = new_state
            rewards[intersection_id] = reward
        
        self.current_states = new_states
        self.episode_step += 1
        
        done = self.episode_step >= self.max_episode_steps
        
        info = {
            'episode_step': self.episode_step,
            'total_vehicles': sum(sum(state.vehicle_counts) for state in new_states.values())
        }
        
        return new_states, rewards, done, info
    
    def _update_state(self, intersection_id, current_state, action):
        """Update intersection state based on action."""
        # Get traffic level from pattern
        pattern_idx = self.episode_step % len(self.traffic_patterns[intersection_id])
        traffic_level = self.traffic_patterns[intersection_id][pattern_idx]
        
        # Simulate traffic flow based on action
        new_vehicle_counts = []
        for i, current_count in enumerate(current_state.vehicle_counts):
            # Traffic arrival
            arrival_rate = traffic_level * 0.5  # Vehicles per step
            arrivals = np.random.poisson(arrival_rate)
            
            # Traffic departure (depends on signal phase)
            if action == i:  # Green light for this direction
                departure_rate = min(current_count, 8)  # Max 8 vehicles can pass
            else:
                departure_rate = 0
            
            new_count = max(0, current_count + arrivals - departure_rate)
            new_vehicle_counts.append(new_count)
        
        # Update other state components
        new_phase = action
        new_phase_time = 0 if action != current_state.current_phase else current_state.phase_time + 1
        
        new_queue_lengths = [max(0, vc - 15) for vc in new_vehicle_counts]
        new_waiting_times = [
            wt + 1 if ql > 0 else 0 
            for wt, ql in zip(current_state.waiting_times, new_queue_lengths)
        ]
        
        # Mock LSTM predictions
        new_predictions = [vc + np.random.normal(0, 1) for vc in new_vehicle_counts]
        
        return RLState(
            vehicle_counts=new_vehicle_counts,
            current_phase=new_phase,
            phase_time=new_phase_time,
            queue_lengths=new_queue_lengths,
            waiting_times=new_waiting_times,
            lstm_predictions=new_predictions
        )
    
    def _calculate_reward(self, current_state, action, new_state):
        """Calculate reward for the action taken."""
        # Reward components
        
        # 1. Throughput reward (vehicles that passed through)
        vehicles_served = sum(
            max(0, curr - new) 
            for curr, new in zip(current_state.vehicle_counts, new_state.vehicle_counts)
        )
        throughput_reward = vehicles_served * 2
        
        # 2. Waiting time penalty
        total_waiting = sum(new_state.waiting_times)
        waiting_penalty = -total_waiting * 0.1
        
        # 3. Queue length penalty
        total_queue = sum(new_state.queue_lengths)
        queue_penalty = -total_queue * 0.5
        
        # 4. Phase change penalty (encourage stability)
        phase_change_penalty = -5 if action != current_state.current_phase else 0
        
        # 5. Minimum green time reward
        min_green_reward = 2 if new_state.phase_time >= 10 else -1
        
        total_reward = (
            throughput_reward + 
            waiting_penalty + 
            queue_penalty + 
            phase_change_penalty + 
            min_green_reward
        )
        
        return total_reward


def train_rl_agent(intersection_ids, episodes=1000, algorithm='dqn'):
    """Train RL agent using the traffic environment."""
    logger = get_logger("RLTraining")
    logger.info(f"Training RL agent with {algorithm} for {episodes} episodes")
    
    # Initialize environment
    env = TrafficEnvironment(intersection_ids)
    
    # Initialize RL coordinator
    config = init_config()
    rl_coordinator = RLCoordinator(intersection_ids, config)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    
    for episode in range(episodes):
        # Reset environment
        states = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Get actions from RL coordinator
            actions = rl_coordinator.get_coordinated_actions(states)
            
            # Take step in environment
            new_states, rewards, done, info = env.step(actions)
            
            # Store experiences and train
            for intersection_id in intersection_ids:
                if intersection_id in rl_coordinator.agents:
                    agent = rl_coordinator.agents[intersection_id]
                    
                    # Store experience
                    agent.remember(
                        states[intersection_id],
                        actions[intersection_id],
                        rewards[intersection_id],
                        new_states[intersection_id],
                        done
                    )
                    
                    # Train agent
                    if len(agent.memory) > agent.batch_size:
                        loss = agent.replay()
                        if loss is not None:
                            training_losses.append(loss)
            
            # Update for next step
            states = new_states
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(training_losses[-100:]) if training_losses else 0
            logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Loss = {avg_loss:.4f}")
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Loss = {avg_loss:.4f}")
    
    training_results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
        'final_avg_reward': np.mean(episode_rewards[-100:]),
        'episodes': episodes,
        'algorithm': algorithm
    }
    
    return rl_coordinator, training_results


def evaluate_rl_agent(rl_coordinator, intersection_ids, test_episodes=100):
    """Evaluate the trained RL agent."""
    logger = get_logger("RLTraining")
    logger.info(f"Evaluating RL agent for {test_episodes} episodes")
    
    env = TrafficEnvironment(intersection_ids)
    evaluator = ModelEvaluator()
    
    test_rewards = []
    test_lengths = []
    
    for episode in range(test_episodes):
        states = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            actions = {}
            for intersection_id in intersection_ids:
                if intersection_id in rl_coordinator.agents:
                    agent = rl_coordinator.agents[intersection_id]
                    action = agent.choose_action(states[intersection_id], training=False)
                    actions[intersection_id] = action
            
            new_states, rewards, done, info = env.step(actions)
            
            states = new_states
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            if done:
                break
        
        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
    
    evaluation_results = {
        'mean_reward': float(np.mean(test_rewards)),
        'std_reward': float(np.std(test_rewards)),
        'max_reward': float(np.max(test_rewards)),
        'min_reward': float(np.min(test_rewards)),
        'mean_episode_length': float(np.mean(test_lengths)),
        'test_episodes': test_episodes
    }
    
    logger.info(f"Evaluation completed. Mean reward: {evaluation_results['mean_reward']:.2f}")
    return evaluation_results


def save_rl_model(rl_coordinator, training_results, evaluation_results):
    """Save the trained RL model and results."""
    logger = get_logger("RLTraining")
    
    models_dir = Path("models/trained")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training results
    results = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'timestamp': timestamp,
        'intersection_ids': list(rl_coordinator.agents.keys())
    }
    
    results_path = models_dir / f"rl_agent_{timestamp}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model configuration (simplified)
    config_path = models_dir / f"rl_agent_{timestamp}_config.json"
    config_data = {
        'intersection_ids': list(rl_coordinator.agents.keys()),
        'timestamp': timestamp,
        'model_type': 'RL_Coordinator'
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"Results saved: {results_path}")
    logger.info(f"Config saved: {config_path}")
    
    return str(results_path), str(config_path)


def plot_training_results(training_results, save_path=None):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RL Agent Training Results', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(training_results['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Moving average of rewards
    window = 100
    if len(training_results['episode_rewards']) >= window:
        moving_avg = pd.Series(training_results['episode_rewards']).rolling(window).mean()
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)
    
    # Training losses
    if training_results['training_losses']:
        axes[1, 0].plot(training_results['training_losses'])
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Episode lengths
    axes[1, 1].plot(training_results['episode_lengths'])
    axes[1, 1].set_title('Episode Lengths')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved: {save_path}")
    
    plt.show()


def main():
    """Main RL training function."""
    parser = argparse.ArgumentParser(description="Train RL Traffic Signal Control Agent")
    
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--algorithm', default='dqn',
                       choices=['dqn', 'double_dqn', 'dueling_dqn'],
                       help='RL algorithm to use')
    parser.add_argument('--intersections', nargs='+', 
                       default=['intersection_1', 'intersection_2'],
                       help='Intersection IDs to train')
    parser.add_argument('--test-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--plot-results', action='store_true',
                       help='Plot training results')
    
    args = parser.parse_args()
    
    # Initialize configuration and logging
    config = init_config()
    setup_logging(config.get_section('logging'))
    logger = get_logger("RLTraining")
    
    try:
        logger.info("Starting RL agent training")
        print(f"🤖 Training RL agent for {len(args.intersections)} intersections...")
        
        # Train RL agent
        rl_coordinator, training_results = train_rl_agent(
            args.intersections, args.episodes, args.algorithm
        )
        
        print("✅ Training completed!")
        
        # Evaluate agent
        print("🎯 Evaluating trained agent...")
        evaluation_results = evaluate_rl_agent(
            rl_coordinator, args.intersections, args.test_episodes
        )
        
        # Save model and results
        print("💾 Saving trained model...")
        results_path, config_path = save_rl_model(
            rl_coordinator, training_results, evaluation_results
        )
        
        # Plot results
        if args.plot_results:
            print("📊 Generating training plots...")
            plot_path = Path("models/trained") / f"rl_training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_training_results(training_results, plot_path)
        
        # Print final results
        print("\n🎉 RL Training Completed!")
        print(f"Results saved: {results_path}")
        print(f"Config saved: {config_path}")
        
        print(f"\n📊 Final Performance:")
        print(f"  Mean Reward: {evaluation_results['mean_reward']:.2f}")
        print(f"  Std Reward: {evaluation_results['std_reward']:.2f}")
        print(f"  Max Reward: {evaluation_results['max_reward']:.2f}")
        print(f"  Final Training Reward: {training_results['final_avg_reward']:.2f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
