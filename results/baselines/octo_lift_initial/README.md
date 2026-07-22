# Initial Octo Lift Baseline

Model: Octo-Small-1.5  
Task: robosuite Lift with Panda  
Instruction: "lift the red cube"  
Episodes: 100  
Seeds: 42–141  
Successes: 1/100  
Success rate: 1.0%  
Wilson 95% CI: 0.18%–5.45%  
Median steady-state latency: 541.67 ms  
Mean steady-state latency: 548.95 ms  
Mean episode length: 396.87 steps  

Integration:
- Zero-shot Octo checkpoint
- Primary and wrist RGB cameras
- Two-frame history
- Four predicted actions executed before replanning
- Lift demonstration action mean/std used for unnormalization
- Panda BASIC composite controller
- CPU JAX inference
