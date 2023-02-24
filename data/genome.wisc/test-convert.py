from Bio.Affy import CelFile
import pandas as pd
import os

# Set the path of the directory containing the .CEL files
cel_dir = "/Users/ezz/fun/microarray-data/genome.wisc/cel-files"

# Loop through the directory and read each .CEL file
for filename in os.listdir(cel_dir):
    if filename.endswith(".CEL"):
        cel_file = os.path.join(cel_dir, filename)
        with open(cel_file, "r") as handle:
            c = CelFile.read(handle)
        
        # Extract expression values from the CelFile object
        exprs = c.intensities
        
        # Convert the expression values to a pandas DataFrame and save as a .csv file
        df = pd.DataFrame(exprs)
        output_file = os.path.splitext(cel_file)[0] + ".csv"
        df.to_csv(output_file, index=False)
