#!/usr/bin/env python3
"""
Background Infinite Recursive Self-Improvement Daemon

This daemon runs continuous RSI cycles in the background, automatically
improving the model over time. It includes:
- Infinite improvement loop
- Automatic checkpoint saving
- Progress logging
- Safe shutdown mechanism
- Performance tracking
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recursive_self_improvement import (
    RecursiveSelfImprovementAgent,
    RSIConfig,
    SafetyConfig,
    ArchitecturalConfig,
    LearningConfig
)
from core.ssm import StateSpaceModel
from experience.experience_buffer import ExperienceBuffer
from env_runner.environment import Environment

# Configuration
DAEMON_LOG_DIR = "rsi_daemon_logs"
DAEMON_CHECKPOINT_DIR = "rsi_daemon_checkpoints"
DAEMON_PID_FILE = "rsi_daemon.pid"
DAEMON_STOP_FILE = "rsi_daemon.stop"

# Pre-trained model path
PRETRAINED_MODEL_PATH = "cartpole_hybrid_real_model.pth"
PRETRAINED_CONFIG = {
    'env_name': 'CartPole-v1',
    'state_dim': 32,
    'hidden_dim': 64,
    'input_dim': 4,
    'output_dim': 4
}

# Global flag for graceful shutdown
should_stop = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global should_stop
    print(f"\n[DAEMON] Received signal {signum}, initiating graceful shutdown...")
    should_stop = True

def setup_logging():
    """Setup logging for daemon"""
    # Create log directory
    os.makedirs(DAEMON_LOG_DIR, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(DAEMON_LOG_DIR, f"rsi_daemon_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("RSI_Daemon")

def check_stop_signal():
    """Check if stop signal file exists"""
    return os.path.exists(DAEMON_STOP_FILE)

def create_pid_file():
    """Create PID file for daemon tracking"""
    with open(DAEMON_PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

def remove_pid_file():
    """Remove PID file on shutdown"""
    if os.path.exists(DAEMON_PID_FILE):
        os.remove(DAEMON_PID_FILE)

def load_initial_model(logger):
    """Load the initial pre-trained model"""
    logger.info("=" * 60)
    logger.info("Loading Initial Model")
    logger.info("=" * 60)
    
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        logger.error(f"Pre-trained model not found: {PRETRAINED_MODEL_PATH}")
        return None, None
    
    try:
        # Initialize model
        model = StateSpaceModel(
            state_dim=PRETRAINED_CONFIG['state_dim'],
            input_dim=PRETRAINED_CONFIG['input_dim'],
            output_dim=PRETRAINED_CONFIG['output_dim'],
            hidden_dim=PRETRAINED_CONFIG['hidden_dim']
        )
        
        # Load weights
        model.load(PRETRAINED_MODEL_PATH)
        model.eval()
        
        # Initialize experience buffer
        experience_buffer = ExperienceBuffer(max_size=10000, device='cpu')
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"State Dim: {PRETRAINED_CONFIG['state_dim']}")
        logger.info(f"Hidden Dim: {PRETRAINED_CONFIG['hidden_dim']}")
        
        return model, experience_buffer
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

def initialize_rsi(model, experience_buffer, logger):
    """Initialize RSI agent"""
    logger.info("=" * 60)
    logger.info("Initializing RSI Agent")
    logger.info("=" * 60)
    
    try:
        # Create environment
        env = Environment(PRETRAINED_CONFIG['env_name'])
        
        # Configure RSI
        rsi_config = RSIConfig(
            num_episodes_quick=10,
            num_episodes_full=20,
            num_meta_tasks_quick=3,
            num_meta_tasks_full=10,
            meta_task_length=50,
            adaptation_steps=5
        )
        
        safety_config = SafetyConfig(
            performance_window=10,
            min_performance_threshold=-500,
            max_emergency_stops=3
        )
        
        arch_config = ArchitecturalConfig(
            state_dim=PRETRAINED_CONFIG['state_dim'],
            hidden_dim=PRETRAINED_CONFIG['hidden_dim']
        )
        
        learn_config = LearningConfig(
            inner_lr=0.01,
            outer_lr=0.001,
            adaptation_steps=5
        )
        
        # Create RSI agent
        rsi_agent = RecursiveSelfImprovementAgent(
            initial_model=model,
            env=env,
            experience_buffer=experience_buffer,
            device='cpu',
            rsi_config=rsi_config,
            safety_config=safety_config,
            arch_config=arch_config,
            learn_config=learn_config
        )
        
        logger.info("✅ RSI Agent initialized successfully")
        return rsi_agent
        
    except Exception as e:
        logger.error(f"Failed to initialize RSI: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_checkpoint(rsi_agent, cycle_num, logger):
    """Save checkpoint of current best model"""
    os.makedirs(DAEMON_CHECKPOINT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"rsi_checkpoint_cycle{cycle_num}_{timestamp}.pth"
    checkpoint_path = os.path.join(DAEMON_CHECKPOINT_DIR, checkpoint_name)
    
    try:
        rsi_agent.model.save(checkpoint_path)
        logger.info(f"✅ Checkpoint saved: {checkpoint_name}")
        
        # Also save as "latest"
        latest_path = os.path.join(DAEMON_CHECKPOINT_DIR, "rsi_latest.pth")
        rsi_agent.model.save(latest_path)
        
        # Save metrics
        metrics_file = os.path.join(DAEMON_CHECKPOINT_DIR, f"metrics_cycle{cycle_num}.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Cycle: {cycle_num}\n")
            f.write(f"Generation: {rsi_agent.generation}\n")
            f.write(f"Reward: {rsi_agent.current_metrics.avg_reward}\n")
            f.write(f"Adaptation Speed: {rsi_agent.current_metrics.adaptation_speed}\n")
            f.write(f"Generalization: {rsi_agent.current_metrics.generalization_score}\n")
            f.write(f"Meta Efficiency: {rsi_agent.current_metrics.meta_efficiency}\n")
            f.write(f"Stability: {rsi_agent.current_metrics.stability_score}\n")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        return False

def run_infinite_rsi_loop(logger):
    """Main infinite RSI loop"""
    logger.info("=" * 60)
    logger.info("Starting Infinite RSI Daemon")
    logger.info("=" * 60)
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Log Directory: {DAEMON_LOG_DIR}")
    logger.info(f"Checkpoint Directory: {DAEMON_CHECKPOINT_DIR}")
    logger.info(f"To stop: Create file '{DAEMON_STOP_FILE}' or send SIGTERM")
    logger.info("=" * 60)
    
    # Load initial model
    model, experience_buffer = load_initial_model(logger)
    if model is None:
        logger.error("Failed to load initial model, exiting")
        return
    
    # Initialize RSI
    rsi_agent = initialize_rsi(model, experience_buffer, logger)
    if rsi_agent is None:
        logger.error("Failed to initialize RSI, exiting")
        return
    
    # Main loop
    cycle = 0
    total_improvements = 0
    best_reward = float('-inf')
    
    logger.info("\n" + "=" * 60)
    logger.info("🚀 INFINITE RSI LOOP STARTED")
    logger.info("=" * 60)
    
    while not should_stop and not check_stop_signal():
        cycle += 1
        
        logger.info("\n" + "=" * 60)
        logger.info(f"RSI Cycle {cycle}")
        logger.info("=" * 60)
        logger.info(f"Current Generation: {rsi_agent.generation}")
        logger.info(f"Current Reward: {rsi_agent.current_metrics.avg_reward:.2f}")
        logger.info(f"Best Reward So Far: {best_reward:.2f}")
        logger.info(f"Total Improvements: {total_improvements}")
        
        try:
            # Run one improvement cycle
            logger.info("\n🔄 Running improvement cycle...")
            improved = rsi_agent.attempt_self_improvement()
            
            if improved:
                total_improvements += 1
                new_reward = rsi_agent.current_metrics.avg_reward
                
                logger.info(f"✅ Improvement found!")
                logger.info(f"   New Reward: {new_reward:.2f}")
                logger.info(f"   Adaptation Speed: {rsi_agent.current_metrics.adaptation_speed:.2f}")
                logger.info(f"   Stability: {rsi_agent.current_metrics.stability_score:.2f}")
                
                # Save checkpoint if new best
                if new_reward > best_reward:
                    best_reward = new_reward
                    logger.info(f"🎉 NEW BEST REWARD: {best_reward:.2f}")
                    save_checkpoint(rsi_agent, cycle, logger)
            else:
                logger.info("⚠ No improvement found in this cycle")
            
            # Log current status
            logger.info(f"\nCurrent Status:")
            logger.info(f"  Generation: {rsi_agent.generation}")
            logger.info(f"  Reward: {rsi_agent.current_metrics.avg_reward:.2f}")
            logger.info(f"  State Dim: {rsi_agent.arch_config.state_dim}")
            logger.info(f"  Hidden Dim: {rsi_agent.arch_config.hidden_dim}")
            logger.info(f"  Emergency Stops: {rsi_agent.safety_monitor.emergency_stop_count}")
            logger.info(f"  Checkpoints: {len(rsi_agent.checkpoint_manager.checkpoints)}")
            
            # Periodic checkpoint (every 10 cycles)
            if cycle % 10 == 0:
                logger.info("\n📦 Periodic checkpoint...")
                save_checkpoint(rsi_agent, cycle, logger)
            
            # Check safety
            if rsi_agent.safety_monitor.emergency_stop_count >= rsi_agent.safety_config.max_emergency_stops:
                logger.warning("⚠ Maximum emergency stops reached, pausing for safety")
                logger.info("Waiting 60 seconds before continuing...")
                time.sleep(60)
                rsi_agent.safety_monitor.emergency_stop_count = 0  # Reset
            
        except Exception as e:
            logger.error(f"❌ Error in cycle {cycle}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("Waiting 30 seconds before retry...")
            time.sleep(30)
        
        # Small delay between cycles
        time.sleep(1)
    
    # Shutdown
    logger.info("\n" + "=" * 60)
    logger.info("🛑 RSI Daemon Shutting Down")
    logger.info("=" * 60)
    logger.info(f"Total Cycles: {cycle}")
    logger.info(f"Total Improvements: {total_improvements}")
    logger.info(f"Best Reward: {best_reward:.2f}")
    logger.info(f"Final Generation: {rsi_agent.generation}")
    
    # Save final checkpoint
    logger.info("\n📦 Saving final checkpoint...")
    save_checkpoint(rsi_agent, cycle, logger)
    
    logger.info("\n✅ Daemon shutdown complete")

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup logging
    logger = setup_logging()
    
    # Create PID file
    create_pid_file()
    
    try:
        # Run infinite loop
        run_infinite_rsi_loop(logger)
    finally:
        # Cleanup
        remove_pid_file()
        if os.path.exists(DAEMON_STOP_FILE):
            os.remove(DAEMON_STOP_FILE)

if __name__ == "__main__":
    main()

