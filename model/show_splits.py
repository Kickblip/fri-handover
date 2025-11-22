"""
Quick script to show which stems are in train/validation/test splits.
Use test or validation stems for inference to evaluate on unseen data.
"""
from .data import list_stems, split_stems

def main():
    stems = list_stems()
    if not stems:
        print("No stems found!")
        return
    
    tr, va, te = split_stems(None)
    
    print(f"Total stems: {len(stems)}")
    print(f"\nData splits:")
    print(f"  Train:      {tr} ({len(tr)} stems)")
    print(f"  Validation: {va} ({len(va)} stems)")
    print(f"  Test:       {te} ({len(te)} stems)")
    
    print(f"\n✅ Use these for inference (not seen during training):")
    if te:
        print(f"  Test stems: {te}")
    if va:
        print(f"  Validation stems: {va}")
    
    if not te and not va:
        print("  ⚠️  No test/validation stems available. All data was used for training.")

if __name__ == "__main__":
    main()

