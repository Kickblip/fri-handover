import numpy as np
import os
import pandas as pd


def synchronize(boxDirPath, handsDirPath, pathToSave):
    
    for filename in os.listdir(boxDirPath):
        if not filename.endswith(".csv"):
            continue

        boxPath = os.path.join(boxDirPath, filename)
        handsFilename = filename[:-len("vertices.csv")] + "world.csv"
        handsPath = os.path.join(handsDirPath, handsFilename)

        boxCSV = pd.read_csv(boxPath)    
        handsCSV = pd.read_csv(handsPath)


        combined = pd.concat(
            [handsCSV, boxCSV.iloc[:, 3:]], 
            axis=1
        )

        outputFilename = filename[:-len("vertices.csv")] + "sync.csv"
        outputPath = os.path.join(pathToSave, outputFilename)

        combined.to_csv(outputPath, index=False)


def main():
    iterations = 1
    handsCSVs = '''<Path to hands coordinates CSV goes here>''' 
    boxCSVs = '''<Path to box coordinates CSV goes here>'''
    pathToSave = '''<Path where the output file should be stored>'''

    synchronize(handsCSVs, boxCSVs, pathToSave)


if __name__ == "__main__":
    main()
    