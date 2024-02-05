import json
import random
import argparse as ap

# grabs a random prompt from <file> and formats it with ATIREsearch tags

def grab_prompt(prompts, dim = 1, tag = "paraphrased"):
    i = random.randint(0, len(prompts)-1)
    if dim == 2:
        j = random.randint(0, len(prompts[i])-1)

        prompt = prompts[i][j]
    else:
        prompt = prompts[i]
    return prompt[tag]

def format_query(prompt):
    tag = "ATIREsearch"
    q = "query"

    return f"<{tag}><{q}>{prompt}</{q}></{tag}>"


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="extract a random prompt from a given json file of llama outputs")
    parser.add_argument("filename", help="json file to extract prompt")
    parser.add_argument("--output", "-o", default="_", help="output file (optional)")
    parser.add_argument("-n", default=1, type=int, help="number of prompts to grab")
    parser.add_argument("--tag", "-t", default="paraphrased", help="tag to use when retrieving prompts from file")

    args = parser.parse_args()

    f = open(args.filename)
    prompts = json.loads(f.read())
    #prompts[0] is a list... prompts[x][y] is a dict
    f.close()

    if args.output != "_":
        f = open(args.output, "w")

        if args.n == -1:
            # run for *every* prompt in dataset
            pass
        else:
            for i in range(args.n):
                p = grab_prompt(prompts)
                prompt = format_query(p)
                f.write(prompt)

        f.close()
    else:
        for i in range(args.n):
            p = grab_prompt(prompts)
            prompt = format_query(p)
            print(prompt)
    
    
    

    