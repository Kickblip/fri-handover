from playsound3 import playsound
import os

def main():
    filename = "beep.mp3"
    audio_path = os.path.join(os.path.dirname(__file__), '..', 'audio', f'{filename}')
    playsound(audio_path)

if __name__ == "__main__":
    main()