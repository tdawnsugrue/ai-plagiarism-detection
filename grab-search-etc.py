"""

1. Open file, grab x or all paraphrased
2. format everything
3. THRU THIS SCRIPT, run atire search on each query, generating an output file for each
4. grab the top hit from search results
5. print the original prompt, the paraphrased version, and the top hit

"""
import json
import os
import sys
import numpy as np
from reformat_for_atire import format_for_query
from extract_features import get_top_n_terms
from get_search_results import get_top_hits

def error_checks():
    try : os.listdir("ATIRE").index("bin")
    except : 
        print("ERROR: could not find ATIRE/bin directory")
        exit()
    try : os.listdir("ATIRE/bin").index("atire")
    except:
        print("ERROR: could not find ATIRE executable")
        exit()
    try : os.listdir("sherlock").index("sherlock")
    except:
        print("ERROR: Sherlock executable not found in subdirectory")
        exit()

def debug_print(item: dict, result: str):
    print("="*6, "ORIGINAL", "="*6)
    print(item['assistant'])
    print("="*20)
    print("="*6, "PARAPHRASED", "="*6)
    print("="*20)
    print(item['paraphrased'])
    print("="*20)
    print("="*8, "HIT", "="*8)
    print(result)


# === LOOK HERE
def read_and_format(content: dict, tag="paraphrased", outfile="para_queries.txt"):
    file = open(outfile, "w")
    try:
        # if multiple queries
        type(content[0]) == dict
        for item in content:
            query_terms = get_top_n_terms(item[tag], -1, True)
            formatted = format_for_query(query_terms)
            file.write(formatted)
    except:
        # if a single dict
        query_terms = get_top_n_terms(content[tag], -1, True)
        if len(query_terms) == 0: return False
        formatted = format_for_query(query_terms)
        file.write(formatted)
    file.close()
    return True


def output_for_sherlock(query: str, hits: list):
    d = "for_sherlock"
    f = open(f"{d}/query.txt", "w")
    f.write(query)
    f.close()

    for i in range(len(hits)):
        f = open(f"{d}/hit_{i}.txt", "w")
        f.write(hits[i])
        f.close()

#os.system("../bin/index -N10000 -rtrec generated-chat-atire.xml")
#we may want to separate this and do it a query at a time (so we retain doc info...?)

# SEARCH! AND DETECT!
# convert paradata to np array
def search_and_detect(paradata, verbose = False, atire_stats = True, sherlock_stats = False, gen_index = False):
    f = open("generated-flat.json")
    prompts = json.loads(f.read())[0]
    f.close()
    
    paradata = np.array(paradata)

    correct_count = 0
    invalid_count = 0
    total_count = 0

    for item in paradata:
        if len(item['assistant']) < 50 or len(item['paraphrased']) < 50:
            # ignore too-short or broken entries
            continue
        read_and_format(item)
        os.system("./ATIRE/bin/atire -k10 -q para-queries.txt > 000.txt")
        search_results = get_top_hits("000.txt", prompts, -1)
        output_for_sherlock(item['paraphrased'], search_results)
        if verbose and False:
            print("query:", item['user'].strip())
            os.system("grep -E 'query' tmp_sher_out.txt")
        os.system("./sherlock/sherlock -e .txt -t 1 * > tmp_sher_out.txt")
        
        correct_count += atire_found_correct(item, search_results)
        total_count += 1
        if verbose:
            if total_count % 100 == 0:
                print("Counted ", total_count)

    print("Atire accuracy:", round(correct_count / total_count, 2), f"% ({correct_count} of {total_count})")

def atire_found_correct(data, hits):
    original = data['assistant']
    for h in hits:
        if original == h : return True
    return False

# text ver
def test_if_correct(data, hits):
    original = data['assistant']
    print("=====\nORIGINAL=====\n", original, "\n========\nTOP HIT==========\n", hits[0])
    print("Matching = ", original == hits[0])

if __name__ == "__main__":
    error_checks()

    f = open("output-with-paraphrasing.json")
    paradata = json.loads(f.read())
    f.close()

    search_and_detect(paradata, verbose=True)

"""
Desired output:

[Original Prompt] || ID = [#]

===
Llama Output
===

[llama output]

===
Paraphrased output
===

Para'd

===
ATIRE retrieved + ret. id
===

Retrieved


=========
Statistics 
=========

% pulled correct (top-1)
% pulled correct (top-10)

"""

