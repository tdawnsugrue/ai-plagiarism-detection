import json
import argparse as ap

# convert generated chat to single file multiple doc tags
"""
example:
    <DOC>
    <DOCNO> [docid] </DOCNO>
    [text goes here]
    </doc>
"""
def format_for_index(filename = "generated-7b-chat.json", tag = "assistant", 
                     outfile = "generated-chat-atire.xml"):
        f = open(filename)
        genchat = json.loads(f.read())
        f.close()

        f = open(outfile, "w")
        c = 1
        for subset in genchat:
            for prompt in subset:
                f.write(f"<DOC>\n<DOCNO>{c}</DOCNO>\n{prompt[tag]}\n</DOC>\n")
                c += 1

        f.close()
        
        print(f"rewrote {c} prompts")

def format_query_for_index(query: str, outfile):
    outfile.write(f"<DOC>\n<DOCNO>1</DOCNO>\n{query}\n</DOC>")

# grab_prompt.py already does this...
def format_json_for_query(filename = "output-with-paraphrasing.json", tag = "paraphrased",
                     outfile = "para-queries.txt"):
    f = open(filename)
    data = json.loads(f.read())
    f.close()

    astag = "ATIREsearch"
    q = "query"

    f = open(outfile, "w")
    for item in data:
        f.write(f"<{astag}><{q}>{item[tag]}</{q}></{astag}>\n")

    f.close()
    print(f"wrote queries to {outfile}")

# formats any text for a query
def format_for_query(content) -> str:
    astag = "ATIREsearch"
    q = "query"

    queries = []
    if type(content) == str:
        return f"<{astag}><{q}>{content}</{q}></{astag}>\n"
    elif type(content) == list:
        for item in content:
            if type(item) != str:
                print("Type Error: could not parse query")
                return
            queries.append(f"<{astag}><{q}>{item}</{q}></{astag}>\n")
        return queries   
    else:
        print("ERROR: query content should be of type list or str")
        return
    
    

if __name__ == "__main__":
    argparser = ap.ArgumentParser

    argparser.add_argument("mode", help="query|index - determines whether to format in trec <DOC><DOCID> or <ATIREsearch><query> tags")

    pars = argparser.parse_args

    if pars.mode == "query":
        format_json_for_query()
    elif pars.mode == "index":
         format_for_index()
    else:
         print("invalid formatting mode.")

    # f = open("generated-7b-chat.json")
    # genchat = json.loads(f.read())
    # f.close()

    # f = open("generated-chat-atire.xml", "w")
    # c = 1
    # for subset in genchat:
    #     for prompt in subset:
    #         f.write(f"<DOC>\n<DOCNO>{c}</DOCNO>\n{prompt['assistant']}\n</DOC>\n")
    #         c += 1

    # f.close()
    
    # print(f"rewrote {c} prompts")
    