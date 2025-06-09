import os

base_path = "../data/raw"
out_tpath = "../data/interm/combined_SC_data.txt"


with open(out_tpath, "w", encoding="utf-8") as outfile:

    for filename in os.listdir(base_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(base_path, filename)

            with open(file_path, "r", encoding="utf-8") as infile:
                contents = infile.read()
                outfile.write(contents.strip() + "\n")

            print(f"Processed {filename}")
