import time
from KinectClient import KinectClient
from PrompterClient import PrompterClient
from typing import Tuple


RED = (180, 0, 0)
GREEN = (10, 127, 0)
BLUE = (30, 58, 138)
GRAY = (34, 34, 34)
RECORDING_SECONDS=4

def get_names() -> Tuple[str, str]:
    user1 = input("Enter name for User 1: ").strip().lower()
    user2 = input("Enter name for User 2: ").strip().lower()
    return user1, user2

def main():
    
    confirm = " "
    while confirm != 'y':
        print("Hello! Please enter your actual names.")
        user1, user2 = get_names()
        n_trials = int(input("Enter number of trials (default 10): ") or 10)
        offset = int(input("How many trials have you already recorded? (default 0): ") or 0)
        print()
        print(f"User 1: {user1}, User 2: {user2}")
        print(f"Number of trials: {n_trials}")
        print(f"Starting number: {offset + 1}\n")
        confirm = input("Are these settings correct? (y/n): ").strip().lower()

    print("Starting recording sequence\n")
    prompter = PrompterClient()
    prompter.show("Get into position!", GRAY); time.sleep(5)
    prompter.show("Starting in 3", RED); time.sleep(1)
    prompter.show("Starting in 2", RED); time.sleep(1)
    prompter.show("Starting in 1", RED); time.sleep(1)
    
    kinect_client = KinectClient()
    try:             
        for trial in range(1, n_trials + 1):
            print(f"Trial {trial} of {n_trials}")
            prompter.show(f"Trial {trial} of {n_trials}", RED); time.sleep(1.5)

            if trial % 2 != 0:
                prompter.show(f"{user1} hands to {user2}", RED); time.sleep(0.5)
            else:
                prompter.show(f"{user2} hands to {user1}", RED); time.sleep(0.5)
                
            prompter.show("Start!", GREEN)
            kinect_client.start_recording(path=f"{offset + trial}_video.mkv", n_seconds=RECORDING_SECONDS)
            prompter.show("Done!", BLUE); time.sleep(0.5)
        
        prompter.show("Recording complete", GRAY); time.sleep(1)
        print("Recording complete. Thank you!")
        prompter.close()
        
    except KeyboardInterrupt:
        kinect_client.close()
        prompter.close()

if __name__ == "__main__":
    main()