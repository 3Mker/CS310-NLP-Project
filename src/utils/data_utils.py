import os, json


def load_chinese_qwen2_dataset(data_path, data_type):
    """Load Chinese Qwen-2 generated and human data, returning texts and binary labels."""
    prompts, results, labels = {}, {}, {}
    new_id = 0
    # load generated (model) data as label 1
    gen_path = os.path.join(data_path, 'generated/zh_qwen2')
    human_path = os.path.join(data_path, 'human/zh_unicode')
    # load generated data as label 1
    for fname in os.listdir(gen_path):
        if fname.endswith('.json'):
            if_type_flag = False
            for type in data_type:
                if type in fname:
                    if_type_flag = True
                    break
            if not if_type_flag:
                continue
            with open(os.path.join(gen_path, fname), encoding='utf-8') as f:
                data = json.load(f)
            input_prompt = {}
            output_result = {}
            for line in data:
                if line == 'input':
                    for id in data[line]:
                        input_prompt[id] = data[line][id]
                elif line == 'output':
                    for id in data[line]:
                        output_result[id] = data[line][id]
            for id in input_prompt:
                if id in output_result:
                    prompts[new_id] = input_prompt[id]
                    results[new_id] = output_result[id]
                    labels[new_id] = 1
                    new_id += 1
    # load human data as label 0
    for fname in os.listdir(human_path):
        if fname.endswith('.json'):
            if_type_flag = False
            for type in data_type:
                if type in fname:
                    if_type_flag = True
                    break
            if not if_type_flag:
                continue
            with open(os.path.join(human_path, fname), encoding='utf-8') as f:
                data = json.load(f)
            input_prompt = {}
            output_result = {}
            id = 0
            for line in data:
                input_prompt[id] = line['input']
                output_result[id] = line['output']
                id += 1
            for id in input_prompt:
                if id in output_result:
                    prompts[new_id] = input_prompt[id]
                    results[new_id] = output_result[id]
                    labels[new_id] = 0
                    new_id += 1
    # Combine prompts, results, and labels into a single dic
    print(f"Loaded {new_id} records from {gen_path} and {human_path}.")
    records = {}
    for i in range(new_id):
        records[i] = {
            'prompt': prompts[i],
            'result': results[i],
            'label': labels[i]
        }
    counter = 0
    for i in range(new_id):
        if records[i]['label'] == 0:
            counter += 1
    print(f"Loaded {counter} human records and {new_id - counter} generated records.")
    return records

def load_english_ghostbuster_dataset(data_path, data_type):
    """Load English Ghostbuster dataset, returning texts and binary labels."""
    prompts, results, labels = {}, {}, {}
    new_id = 0
    types = os.listdir(data_path)
    for type in types:
        if type not in data_type:
            continue
        if type != 'reuter':
            type_path = os.path.join(data_path, type)
            prompts_temp = {}
            print(f"Loading {type} dataset from {type_path}...")
            # process prompts first
            for folder in os.listdir(type_path):
                if 'prompts' in folder:
                    folder_path = os.path.join(type_path, folder)
                    for txt_file in os.listdir(folder_path):
                        if txt_file.endswith('.txt'):
                            txt_id = int(txt_file.split('.')[0])
                            one_line = ''
                            with open(os.path.join(folder_path, txt_file), encoding='utf-8') as f:
                                lines = f.readlines()
                                one_line = ''.join([line.strip() for line in lines])
                            prompts_temp[txt_id] = one_line

            for folder in os.listdir(type_path):
                folder_path = os.path.join(type_path, folder)
                if 'prompts' in folder:
                    continue
                elif 'human' in folder:
                    for txt_file in os.listdir(folder_path):
                        if txt_file.endswith('.txt'):
                            txt_id = int(txt_file.split('.')[0])
                            one_line = ''
                            with open(os.path.join(folder_path, txt_file), encoding='utf-8') as f:
                                lines = f.readlines()
                                one_line = ''.join([line.strip() for line in lines])
                            prompts[new_id] = prompts_temp[txt_id]
                            results[new_id] = one_line
                            labels[new_id] = 0
                            new_id += 1
                # ignore hidden folders
                elif folder.startswith('.'):
                    continue
                else:
                    for txt_file in os.listdir(folder_path):
                        if txt_file.endswith('.txt'):
                            txt_id = int(txt_file.split('.')[0])
                            one_line = ''
                            with open(os.path.join(folder_path, txt_file), encoding='utf-8') as f:
                                lines = f.readlines()
                                one_line = ''.join([line.strip() for line in lines])
                            prompts[new_id] = prompts_temp[txt_id]
                            results[new_id] = one_line
                            labels[new_id] = 1
                            new_id += 1
        else:
            # process reuter dataset
            type_path = os.path.join(data_path, type)
            print(f"Loading {type} dataset from {type_path}...")
            for folder in os.listdir(type_path):
                # no prompts in reuter dataset
                folder_path = os.path.join(type_path, folder)
                # ignore hidden folders
                if folder.startswith('.'):
                    continue
                for name in os.listdir(folder_path):
                    name_path = os.path.join(folder_path, name)
                    for txt_file in os.listdir(name_path):
                        if txt_file.endswith('.txt'):
                            one_line = ''
                            with open(os.path.join(name_path, txt_file), encoding='utf-8') as f:
                                lines = f.readlines()
                                one_line = ''.join([line.strip() for line in lines])
                            prompts[new_id] = 'Reported by: ' + name + '.'
                            results[new_id] = one_line
                            if 'human' in folder:
                                labels[new_id] = 0
                            else:
                                labels[new_id] = 1
                            new_id += 1
    # Combine prompts, results, and labels into a single dic
    print(f"Loaded {new_id} records from {data_path}.")
    records = {}
    for i in range(new_id):
        records[i] = {
            'prompt': prompts[i],
            'result': results[i],
            'label': labels[i]
        }
    counter = 0
    for i in range(new_id):
        if records[i]['label'] == 0:
            counter += 1
    print(f"Loaded {counter} human records and {new_id - counter} generated records.")
    return records


if __name__ == "__main__":
    # Example usage
    data_path = "face2_zh_json"
    texts, labels = load_chinese_qwen2_dataset(data_path)
    print(f"Loaded {len(texts)} texts and {len(labels)} labels.")
    print(texts[0])
    print(labels[0])