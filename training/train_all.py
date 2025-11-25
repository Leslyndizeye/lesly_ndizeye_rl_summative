"""
Train all algorithms sequentially
"""

import subprocess
import sys


def main():

    print("TRAINING ALL ALGORITHMS FOR PHARMASTOCK")
    
    scripts = [
        'training/dqn_training.py',
        'training/ppo_training.py',
        'training/a2c_training.py',
        'training/reinforce_training.py'
    ]
    
    for script in scripts:
        print(f"\n\nRunning: {script}")
        print("="*70)
        subprocess.run([sys.executable, script])
    
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
