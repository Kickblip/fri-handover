import time
from helpers import get_names, play
from KinectClient import KinectClient

def main():
    kinect_client = KinectClient()
    
    confirm = ""
    while confirm[0] != 'y':
        print("Hello! Please enter your actual names.")
        user1, user2 = get_names()
        n_trials = int(input("Enter number of trials (default 10): ") or 10)
        print()
        print(f"REMEMBER || User 1: {user1}, User 2: {user2}")
        print(f"Number of trials: {n_trials}\n")
        confirm = input("Are these settings correct? (y/n): ").strip().lower()

    print("Starting recording sequence\n")
    time.sleep(2)
    play("starting.mp3")
    time.sleep(5)
    play("ready-set.mp3")
    time.sleep(0.5)
    play("beep.mp3")
    
    for trial in range(1, n_trials + 1):
        print(f"Trial {trial} of {n_trials}")
        
        play("two-one.mp3")
        time.sleep(1)
        play("beep.mp3")
        
        if trial % 2 != 0:
            play("one-two.mp3")
            kinect_client.start_recording(path=f"sequences/{trial}/rgb/{user1}_{user2}.mkv", n_seconds=5)
        else:
            play("two-one.mp3")
            kinect_client.start_recording(path=f"sequences/{trial}/rgb/{user2}_{user1}.mkv", n_seconds=5)

        play("beep.mp3")
    
    play("complete.mp3")
    print("Recording complete. Thank you!")

if __name__ == "__main__":
    main()