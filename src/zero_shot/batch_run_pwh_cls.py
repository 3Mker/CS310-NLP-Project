import os
import subprocess

# Define the base directory containing the FFT data
fft_data_dir = "fft_output/cn"

# Define the output directory for results
results_dir = "fft_results/pwh_cls/cn"
os.makedirs(results_dir, exist_ok=True)

# Iterate through the genres (e.g., essay, reuter, wp)
for genre in os.listdir(fft_data_dir):
    genre_path = os.path.join(fft_data_dir, genre)
    if not os.path.isdir(genre_path):
        continue

    # Iterate through the model types (e.g., gpt, human)
    if fft_data_dir == 'fft_output/english':
        for model_type in os.listdir(genre_path):
            model_type_path = os.path.join(genre_path, model_type)
            if not os.path.isdir(model_type_path):
                continue
            if model_type == 'human':
                continue
            print(f"Processing genre: {genre}, model type: {model_type}")
            files = os.listdir(model_type_path)
            for file in files:
                file_path = os.path.join(model_type_path, file)
                human_file_path = os.path.join(genre_path, 'human', file)
                print(f"Checking file: {file_path}, human file: {human_file_path}")
                

                # Define the output file path
                output_file = os.path.join(results_dir, f"{genre}_{model_type}_{file}.pwh_cls.txt")

                # Run the PWH classification script
                command = [
                    "python", "FourierGPT/run_pwh_cls.py",
                    "--model", file_path,
                    "--human", human_file_path
                ]
                with open(output_file, "w") as of:
                    subprocess.run(command, stdout=of, stderr=subprocess.STDOUT, check=True)
                print(f"Output saved to: {output_file}")
    else:
        files = os.listdir(genre_path)
        for file in files:
            file_path = os.path.join(genre_path, file)
            if not os.path.isfile(file_path):
                continue
            if file.__contains__('0'):
                continue
            print(f"Processing file: {file_path}")

            # Define the output file path
            output_file = os.path.join(results_dir, f"{genre}_{file}.pwh_cls.txt")

            # 吧filepath的1换成'0'
            human_file_path = file_path.replace('1', '0')
            print(f"Human file path: {human_file_path}")

            # Run the PWH classification script
            command = [
                "python", "FourierGPT/run_pwh_cls.py",
                "--model", file_path,
                "--human", human_file_path
            ]
            with open(output_file, "w") as of:
                subprocess.run(command, stdout=of, stderr=subprocess.STDOUT, check=True)
            print(f"Output saved to: {output_file}")