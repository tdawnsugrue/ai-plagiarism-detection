# Script to use the msmarco dataset with chat completion
import json
from typing import List, Optional
import fire
from llama import Llama, Dialog

#from .format-data import format_data; formats msmarco
def format_data(filename: str, limit: int = 0, write_path: str = ""):
    f = open(filename)

    data = []

    line = f.readline()
    count = 0 if limit else -1
    total = 0
    while line and count < limit:
        start = line.find(" ") + 1

        data.append(
            [{"role" : "user",
              "content" : line[start:].strip()}]
        )
        if limit:
            count += 1
        total += 1
        line = f.readline()
    print("Added", total, "queries")
    f.close()

    #print(data)
    return data

# formats outputs from llama-2-chat to something useful
def format_outputs(dialogs, results):
    out = []
    for dialog, result in zip(dialogs, results):
        prompt = {
            "user" : "",
            "assistant" : result['generation']['content']
        }
        for msg in dialog:
            prompt['user'] += msg['content'] + '\n'
        out.append(prompt)

    return out


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    
    generator = Llama.build(ckpt_dir=ckpt_dir, 
                            tokenizer_path=tokenizer_path,
                            max_seq_len=max_seq_len,
                            max_batch_size=max_batch_size,)
    
    dialogs: List[Dialog] = format_data("queries.eval.tsv")

    # len(dialogs)
    for i in range(20720, 20741):
        try:
            results = generator.chat_completion(
                dialogs[i:i+1],  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            
        except RuntimeError as e:
            print(f"Runtime error occurred at line {i}. 'prob tensor contains either inf, nan, or element <0\nmsmarco entry:\n")
            print(dialogs[i])
    f = open("bad_llama_subset.json", "w")
    output = format_outputs(dialogs, results)
    json.dump(output, f)
    f.close()



if __name__ == "__main__":
    fire.Fire(main)