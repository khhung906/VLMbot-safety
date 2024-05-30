import torch

TRAIN_COLORS = ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']
EVAL_COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'white']

def encode_text(model, tokenizer, sentence):
    with torch.no_grad():
        inputs = tokenizer(sentence, return_tensors="pt")
        emb = model(**inputs).pooler_output[0]

    return emb
  
def test_rewrite(model, tokenizer, atk_sentence, all_tasks, target_idx): 
    atk_emb = encode_text(model, tokenizer, atk_sentence)
    target_embs = [encode_text(model, tokenizer, task) for task in all_tasks]

    dot_products = [atk_emb.dot(target_emb.T).item() for target_emb in target_embs]

    # Find the index of the maximum dot product
    max_index = dot_products.index(max(dot_products))

    print(f"[{max_index==target_idx}] nearest task: {all_tasks[max_index]}; attack sentence: {atk_sentence}")

    return max_index