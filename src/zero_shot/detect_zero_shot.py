import argparse
from utils.data_utils import load_dataset, preprocess_text

def zero_shot_detect(texts, method='detectgpt'):
    """Apply zero-shot detection method to a list of texts and return detection scores."""
    # TODO: implement specific zero-shot detection logic for methods like Fast-DetectGPT, FourierGPT, GPT-who, etc.
    # For now, return dummy scores
    return [0.5 for _ in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--method', type=str, default='detectgpt', help='zero-shot detection method')
    parser.add_argument('--output_path', type=str, default='results/zero_shot/scores.csv')
    args = parser.parse_args()

    texts, _ = load_dataset(args.data_path)
    texts = [preprocess_text(t) for t in texts]

    scores = zero_shot_detect(texts, method=args.method)

    # Save scores
    import csv
    with open(args.output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text_index', 'score'])
        for idx, score in enumerate(scores):
            writer.writerow([idx, score])

    print(f"Zero-shot detection scores saved to {args.output_path}")

if __name__ == '__main__':
    main()