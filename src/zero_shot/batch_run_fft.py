import os
import subprocess

def get_all_files_in_subdirectories(directory):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

# Define the base directory containing the NLL data
# data_dir = "ghostbuster-data_reformed"
data_dir = "results_nll"
out_dir = "fft_output/cn"
# Define the normalization methods and value extraction methods
normalization_methods = ["zscore", "logzs", "minmax"]
value_methods = ["norm", "real", "imag"]

# Iterate through the first-level directories (e.g., essay, wp, reuter)
for data_type in os.listdir(data_dir):
    data_type_path = os.path.join(data_dir, data_type)
    if not os.path.isdir(data_type_path):
        continue
    print(f"Processing data type: {data_type}")
    # Iterate through the second-level directories (e.g., claude, gpt, human)
    count = 0
    if data_dir == 'ghostbuster-data_reformed':
        for model_type in os.listdir(data_type_path):
            model_type_path = os.path.join(data_type_path, model_type)
            if not os.path.isdir(model_type_path):
                continue
            print(f"Processing model type: {model_type}")
            if data_type != 'reuter':
                all_files = os.listdir(model_type_path)
                new_all_files = []
                for file in all_files:
                    if file.endswith("combined-ada.txt") or file.endswith("combined-davinci.txt"):
                        new_all_files.append(os.path.join(model_type_path, file))
                all_files = new_all_files
            else:
                all_files = get_all_files_in_subdirectories(model_type_path)
                # 并且要把所有的combine-ada.txt合并为一个
                combined_ada_file = os.path.join(model_type_path, "final-combined-ada.txt")
                with open(combined_ada_file, 'w') as outfile:
                    for file_name in all_files:
                        if file_name.endswith("combined-ada.txt"):
                            with open(file_name) as infile:
                                outfile.write(infile.read())
                combined_davinvi_file = os.path.join(model_type_path, "final-combined-davinci.txt")
                with open(combined_davinvi_file, 'w') as outfile:
                    for file_name in all_files:
                        if file_name.endswith("combined-davinci.txt"):
                            with open(file_name) as infile:
                                outfile.write(infile.read())
                new_all_files = []
                new_all_files.append(combined_ada_file)
                new_all_files.append(combined_davinvi_file)
                all_files = new_all_files
            # Process only the specific files: combined-ada.txt and combined-davinci.txt
            print(f"Found {len(all_files)} files in {model_type_path}")
            for file_name in all_files:
                if not file_name.endswith("combined-ada.txt") and not file_name.endswith("combined-davinci.txt"):
                    continue
                count += 1
                print(f"Processing file: {file_name}")
                input_file = file_name
                if not os.path.isfile(input_file):
                    continue

                # Generate FFT data for each combination of normalization and value methods
                for norm_method in normalization_methods:
                    for value_method in value_methods:
                        # Construct the output file path
                        output_file = os.path.join(
                            out_dir,
                            data_type,
                            model_type,
                            f"{os.path.basename(file_name)}_{norm_method}_{value_method}.txt"
                        )
                        # make_sure_dir = os.path.join(output_file,data_type, model_type)
                        # make_sure_dir = os.path.dirname(make_sure_dir)
                        # os.makedirs(make_sure_dir, exist_ok=True)
                        print(f"Output file: {output_file}")
                        # Run the FFT processing script
                        command = [
                            "python", "FourierGPT/run_fft.py",
                            "-i", input_file,
                            "-o", output_file,
                            "-p", norm_method,
                            "--value", value_method
                        ]

                        print(f"Processing: {input_file} -> {output_file} with -p {norm_method} --value {value_method}")
                        try:
                            subprocess.run(command, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error processing {input_file} with -p {norm_method} --value {value_method}: {e}")
    else :
        input_file = os.path.join(data_type_path, 'nll_scores_0.txt')
        for norm_method in normalization_methods:
            for value_method in value_methods:
                output_file = os.path.join(out_dir, data_type, f"nll_scores_0_{norm_method}_{value_method}.txt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                command = [
                    "python", "FourierGPT/run_fft.py",
                    "-i", input_file,
                    "-o", output_file,
                    "-p", norm_method,
                    "--value", value_method
                ]
                print(f"Processing: {input_file} -> {output_file} with -p {norm_method} --value {value_method}")
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {input_file} with -p {norm_method} --value {value_method}: {e}")
                count += 1
        input_file = os.path.join(data_type_path, 'nll_scores_1.txt')
        for norm_method in normalization_methods:
            for value_method in value_methods:
                output_file = os.path.join(out_dir, data_type, f"nll_scores_1_{norm_method}_{value_method}.txt")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                command = [
                    "python", "FourierGPT/run_fft.py",
                    "-i", input_file,
                    "-o", output_file,
                    "-p", norm_method,
                    "--value", value_method
                ]
                print(f"Processing: {input_file} -> {output_file} with -p {norm_method} --value {value_method}")
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {input_file} with -p {norm_method} --value {value_method}: {e}")
                count += 1

        # Print the count of processed files for the current model type
    # print(f"Processed {count} files in {model_type_path}")