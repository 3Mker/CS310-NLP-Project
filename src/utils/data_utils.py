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
    return records





if __name__ == "__main__":
    # Example usage
    data_path = "face2_zh_json"
    texts, labels = load_chinese_qwen2_dataset(data_path)
    print(f"Loaded {len(texts)} texts and {len(labels)} labels.")
    print(texts[0])
    print(labels[0])