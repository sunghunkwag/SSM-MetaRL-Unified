# Background Infinite RSI Daemon

## Overview

The RSI Daemon runs **continuous recursive self-improvement cycles in the background**, automatically improving the model over time without user intervention.

## Features

✅ **Infinite Loop** - Runs continuously until stopped
✅ **Automatic Checkpointing** - Saves best models automatically
✅ **Progress Logging** - Detailed logs of all improvements
✅ **Safe Shutdown** - Graceful termination on signal
✅ **Performance Tracking** - Monitors all metrics over time
✅ **Error Recovery** - Continues after failures
✅ **Resource Efficient** - CPU-based, runs in background

## Quick Start

### Start the Daemon

```bash
./rsi_control.sh start
```

Output:
```
🚀 Starting RSI Daemon...
✅ RSI Daemon started successfully (PID: 12345)
📁 Logs: rsi_daemon_logs/
🛑 To stop: ./rsi_control.sh stop
```

### Check Status

```bash
./rsi_control.sh status
```

Output:
```
✅ RSI Daemon is running (PID: 12345)

📊 Latest Status (last 20 lines):
==================================
2025-10-25 16:50:00 - RSI Cycle 15
2025-10-25 16:50:00 - Current Reward: 28.50
2025-10-25 16:50:00 - Best Reward So Far: 32.00
2025-10-25 16:50:00 - Total Improvements: 8
...
```

### View Live Logs

```bash
./rsi_control.sh logs
```

This will tail the log file in real-time, showing all RSI activity.

### Stop the Daemon

```bash
./rsi_control.sh stop
```

Output:
```
🛑 Stopping RSI Daemon (PID: 12345)...
✅ RSI Daemon stopped successfully
```

### Restart the Daemon

```bash
./rsi_control.sh restart
```

## How It Works

### Initialization

1. Loads pre-trained model (`cartpole_hybrid_real_model.pth`)
2. Initializes RSI agent with safety configs
3. Creates log and checkpoint directories
4. Starts infinite improvement loop

### Main Loop

```
While not stopped:
  1. Run one RSI improvement cycle
  2. Evaluate current performance
  3. Test architectural and hyperparameter proposals
  4. Select and apply best improvement
  5. Save checkpoint if new best found
  6. Log all metrics and progress
  7. Sleep 1 second
  8. Repeat
```

### Automatic Checkpointing

The daemon saves checkpoints:
- **On new best reward** - Immediately when performance improves
- **Every 10 cycles** - Periodic backup
- **On shutdown** - Final state before stopping

Checkpoints are saved to: `rsi_daemon_checkpoints/`

### Logging

All activity is logged to: `rsi_daemon_logs/rsi_daemon_YYYYMMDD_HHMMSS.log`

Log includes:
- Cycle number and timestamp
- Current and best rewards
- Improvement details
- Architecture changes
- Safety events
- Error messages

## File Structure

```
rsi_daemon_logs/
├── rsi_daemon_20251025_165000.log
├── rsi_daemon_20251025_170000.log
└── ...

rsi_daemon_checkpoints/
├── rsi_checkpoint_cycle10_20251025_165500.pth
├── rsi_checkpoint_cycle20_20251025_170000.pth
├── rsi_latest.pth
├── metrics_cycle10.txt
├── metrics_cycle20.txt
└── ...

rsi_daemon.pid        # Process ID file
rsi_daemon.stop       # Stop signal file (created by control script)
```

## Configuration

Edit `rsi_daemon.py` to customize:

### RSI Configuration

```python
rsi_config = RSIConfig(
    num_episodes_quick=10,      # Episodes for quick eval
    num_episodes_full=20,       # Episodes for full eval
    num_meta_tasks_quick=3,     # Meta-tasks for quick eval
    num_meta_tasks_full=10,     # Meta-tasks for full eval
    meta_task_length=50,        # Steps per meta-task
    adaptation_steps=5          # Adaptation steps
)
```

### Safety Configuration

```python
safety_config = SafetyConfig(
    performance_window=10,              # History window
    min_performance_threshold=-500,     # Min reward
    max_emergency_stops=3               # Max failures
)
```

### Checkpoint Frequency

```python
# Line 268 in rsi_daemon.py
if cycle % 10 == 0:  # Change 10 to desired frequency
    save_checkpoint(rsi_agent, cycle, logger)
```

## Monitoring

### Real-time Monitoring

```bash
# Terminal 1: Watch status
watch -n 5 './rsi_control.sh status'

# Terminal 2: Follow logs
./rsi_control.sh logs
```

### Check Checkpoints

```bash
ls -lh rsi_daemon_checkpoints/
```

### View Metrics

```bash
cat rsi_daemon_checkpoints/metrics_cycle*.txt
```

## Safety Features

### Emergency Stop

If 3 consecutive cycles fail:
- Daemon pauses for 60 seconds
- Resets emergency counter
- Continues automatically

### Graceful Shutdown

On SIGTERM or stop signal:
- Completes current cycle
- Saves final checkpoint
- Writes shutdown log
- Cleans up PID file

### Error Recovery

On cycle error:
- Logs error details
- Waits 30 seconds
- Continues with next cycle

## Performance

### Resource Usage

- **CPU**: ~50-100% of one core
- **Memory**: ~500MB - 1GB
- **Disk**: Grows with checkpoints (~32KB per checkpoint)

### Improvement Rate

Typical performance:
- **Cycle Duration**: 30-60 seconds
- **Improvements**: ~20-40% of cycles
- **Best Reward**: Increases over time
- **Convergence**: Depends on task complexity

## Troubleshooting

### Daemon Won't Start

```bash
# Check if already running
./rsi_control.sh status

# Check for stale PID file
rm -f rsi_daemon.pid

# Try starting again
./rsi_control.sh start
```

### Daemon Stops Unexpectedly

```bash
# Check logs for errors
./rsi_control.sh logs

# Look for emergency stops
grep "emergency" rsi_daemon_logs/*.log

# Check system resources
top
df -h
```

### No Improvements

This is normal! Not every cycle finds improvements. The daemon will:
- Continue searching
- Try different proposals
- Eventually find improvements

### High CPU Usage

This is expected. To reduce:
- Increase sleep time in main loop
- Reduce `num_episodes_quick`
- Reduce `num_meta_tasks_quick`

## Advanced Usage

### Run in Docker

```bash
docker run -d \
  -v $(pwd):/workspace \
  -w /workspace \
  python:3.11 \
  python3 rsi_daemon.py
```

### Run on Server

```bash
# Start daemon
./rsi_control.sh start

# Detach from terminal
exit

# Later, check status via SSH
ssh user@server "cd /path/to/project && ./rsi_control.sh status"
```

### Multiple Daemons

To run multiple daemons (different environments):

1. Copy `rsi_daemon.py` to `rsi_daemon_env2.py`
2. Change `DAEMON_PID_FILE` and log directories
3. Modify environment name
4. Start both daemons

## Best Practices

### 1. Monitor Regularly

Check status at least once per day:
```bash
./rsi_control.sh status
```

### 2. Backup Checkpoints

Periodically backup the best checkpoints:
```bash
cp rsi_daemon_checkpoints/rsi_latest.pth backups/
```

### 3. Review Logs

Weekly review of logs to understand improvement patterns:
```bash
grep "NEW BEST" rsi_daemon_logs/*.log
```

### 4. Clean Old Logs

Remove old logs to save disk space:
```bash
find rsi_daemon_logs/ -name "*.log" -mtime +30 -delete
```

## Integration with Gradio App

The daemon runs independently, but you can:

1. **Load daemon checkpoints in Gradio**:
   - Copy `rsi_daemon_checkpoints/rsi_latest.pth` to `cartpole_hybrid_real_model.pth`
   - Reload in Gradio interface

2. **Compare performance**:
   - Test original model in Gradio
   - Test daemon-improved model
   - Compare rewards

## FAQ

**Q: How long should I run the daemon?**
A: As long as you want! It will continue improving indefinitely.

**Q: Will it ever stop improving?**
A: Eventually it will converge to a local optimum, but it will keep searching.

**Q: Can I run it on GPU?**
A: Yes, modify `device='cpu'` to `device='cuda'` in `rsi_daemon.py`.

**Q: Is it safe to leave running overnight?**
A: Yes! The daemon includes safety features and automatic checkpointing.

**Q: How do I use the improved model?**
A: Copy `rsi_daemon_checkpoints/rsi_latest.pth` to your desired location.

**Q: Can I pause and resume?**
A: Yes! Stop the daemon, and it will save state. Start again to resume.

## Example Session

```bash
# Start daemon
$ ./rsi_control.sh start
🚀 Starting RSI Daemon...
✅ RSI Daemon started successfully (PID: 12345)

# Check status after 1 hour
$ ./rsi_control.sh status
✅ RSI Daemon is running (PID: 12345)
📊 Latest Status:
  Cycle: 120
  Total Improvements: 45
  Best Reward: 45.80
  Current Reward: 43.20

# View live logs
$ ./rsi_control.sh logs
[Following logs in real-time...]
2025-10-25 17:30:00 - RSI Cycle 121
2025-10-25 17:30:00 - ✅ Improvement found!
2025-10-25 17:30:00 - New Reward: 46.50
2025-10-25 17:30:00 - 🎉 NEW BEST REWARD: 46.50

# Stop daemon after 24 hours
$ ./rsi_control.sh stop
🛑 Stopping RSI Daemon (PID: 12345)...
✅ RSI Daemon stopped successfully

# Check final results
$ cat rsi_daemon_checkpoints/metrics_cycle*.txt | grep "Reward" | tail -1
Reward: 52.30
```

## Conclusion

The RSI Daemon enables **continuous autonomous improvement** of your meta-learning model. Set it and forget it - the model will keep getting better!

For questions or issues, check the logs first, then review this README.

**Happy self-improving!** 🚀

