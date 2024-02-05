import json
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text
    

if __name__ == "__main__":
    dp = DipperParaphraser(model="para-paraphrase-ctx-t5-xxl")

    f = open("generated-7b-chat.json")
    g = f.read()
    genchat = json.loads(g)
    f.close()

    all_dict = []
    
    c = 0
    for subset in genchat:
        for prompt in subset:
            # can modify this later if we're not wanting to paraphrase the entire prompt
            # or want to reduce the length of the query.
            p2 = prompt

            #run dp with default params as per paraphrase_minimal.py
            # max seq len 600
            paraphrased = dp.paraphrase(prompt['assistant'], lex_diversity=60, order_diversity=0,top_p=0.75, do_sample=True, top_k=None, max_length=600)

            p2['paraphrased'] = paraphrased
            all_dict.append(p2)
            c += 1


    print(f"generated {c} prompts in approx {time.process_time()} seconds.")

    output = open("output-with-paraphrasing.json", "w")
    output.write(json.dumps(all_dict))
    output.close()
    # forea prompt (small subset for testing)
    # paraphrase using dipper
    # write to array of dict: original prompt + dipper prompt
    # make a separate prompt of dipper only
    # after all generated: write each array of dict to json file

