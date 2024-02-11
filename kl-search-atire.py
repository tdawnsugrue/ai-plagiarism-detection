# read in paraphrased entries
# for each entry:
#       format to atire trec format
#       run thru the indexer; output index and doclist
#       if these dont have the right info then run the index thru atire_dictionary to get wc
#       do a kl divergence to get the top n terms
#       generate a query with the top n terms
#       run the query through atire and pull documents
#       run a check for the correct document to get % statistics
#       [sherlock stuff once we're happy]
import os
import sys
import json
import math
import numpy as np

from reformat_for_atire import format_for_query, format_query_for_index
from get_search_results import get_top_hits
from grab_search_etc import atire_found_correct, output_for_sherlock

def kl_divergence(doc_freq, doc_length, coll_freq, coll_length):
    px = doc_freq / doc_length
    qx = coll_freq / coll_length
    return px * math.log(px / qx)


def top_n_terms(query, n=5):
    # get paraphrased & WRITE
    p = query['paraphrased']

    g = open("query.xml", "w")
    
    format_query_for_index(p, g)
    g.close()

    # index
    os.system("./ATIRE/bin/index -findex queryindex.aspt -fdoclist querydoclist.aspt query.xml")

    # get wordcounts
    os.system("./ATIRE/bin/atire_dictionary queryindex.aspt > query_dictionary.txt")

    #turn wordcounts for each into something useful
    f = open("query_dictionary.txt")
    query_terms = {}
    for line in f.readlines():
        x = line.strip().split(" ")
        query_terms[x[0]] = int(x[1])
    f.close()

    # remove tags and stuff starting with non-alphanumeric
    taglist = ["DOC", "DOCNO"]
    query_length = 0
    remove = []
    for term in query_terms:
        if not term[0].isalnum() or term in taglist:
            remove.append(term)
        else:
            query_length += query_terms[term]
            

    for key in remove: query_terms.pop(key)
    #print(f"{query_length} terms in query")

    f = open("collection_dictionary.txt")
    collection_vocab = {}
    collection_length = 0
    for line in f.readlines():
        x = line.strip().split(" ")
        collection_vocab[x[0]] = int(x[1])
        collection_length += int(x[1])

    f.close()
    
    zz = 0
    v = []
    divergences = []
    for term in query_terms:
        try:
            kl = kl_divergence(query_terms[term], query_length, collection_vocab[term], collection_length)
        except:
            kl = -1
        v.append(term)
        divergences.append(kl)

    v2 = np.array(v)
    #print(v2)
    #print(v2[np.argsort(divergences)][::-1][:5])
    return v2[np.argsort(divergences)][::-1][:5]

def parse_comparison(line):
    if line.find("query") == -1 : return False

    i = line.find("hit_")+4
    hit_number = int(line[i:i+1])
    sim = line[line.find(";")+1:][line.find(";")+1:].strip()
    sim = int(sim[:sim.find("%")])
    #print(f"Hit: {hit_number}, Similarity: {sim}%")
    return(hit_number, sim)


def parse_sherlock(correct):
    f = open("tmp_sher_out.txt")
    hits = np.array(f.readlines())
    #its = hits[hits.find("query") != -1]
    if hits.shape[0] == 0 : return (-1, 0)
    hits = hits[np.char.find(hits, "query") != -1]
    
    #print(f"{len(hits)} for this query")
    corr = -1
    incorr = []

    for h in hits:
        h_num, sim = parse_comparison(h)
        if h_num == correct:
            corr = sim
            continue
        incorr.append(sim)

    # entries sherlock missed assumed to have similarity of 0%
    for _ in range(10 - len(incorr)):
        incorr.append(0)

    avgsim = np.average(incorr)

    return (corr, avgsim)

def write_atire_results(id: int, file):
    f = open("atire-search-results.txt")
    file.write(f"<RESULT>\n<ID>{id}</ID>\n<ATIRE>\n")
    file.write(f.read())
    file.write(f"</ATIRE></RESULT>")
    f.close()

# === Atire version ===
#   [format]
#   use code for getting term frequencies and total count
#   use code for calculating KL divergence of terms in a document
#   make and perform a query doing this
#   do statistics to check for correct document

# test on a subset for now
f = open("output-with-paraphrasing.json")
query_json = json.loads(f.read())
f.close()

hit_history = open("atire-search-results.txt", "w")

total_queries = 0
atire_correct = 0
short_queries = 0

# OTHER STATS... for later?
atire_top_1_correct = 0
avg_ranking = 0

#Sherlock
correct_percentages = []
incorrect_avg_percentages = []

length_cutoff = 50

id = 0
for query in query_json:
    id += 1
    if len(query['paraphrased']) < length_cutoff or len(query['assistant']) < length_cutoff:
        short_queries += 1
        continue
    t = top_n_terms(query)

    qs = ""
    for w in t : qs += w + " "

    f = open("atire-query.txt", "w")
    f.write(format_for_query(qs))
    f.close()

    os.system("./ATIRE/bin/atire -k10 -nologo -q atire-query.txt > atire-search-results.txt")

    write_atire_results(id, hit_history)

    f = open("generated-flat.json")
    prompts = json.loads(f.read())[0]
    f.close()

    hits = get_top_hits("atire-search-results.txt", prompts)
    total_queries += 1

    output_for_sherlock(query['paraphrased'], hits)
    os.system("./sherlock/sherlock -e .txt -t 1 * > tmp_sher_out.txt")

    val, h = atire_found_correct(query, hits)

    correct_id, avg_inc_id = parse_sherlock(h)
    if correct_id != -1 : 
        correct_percentages.append(correct_id)
        incorrect_avg_percentages.append(avg_inc_id)

    atire_correct += val
    if total_queries % 100 == 0:
        print(f"tested {total_queries} queries", file=sys.stderr)

print(f"total queries tested: {total_queries}\natire found {atire_correct/total_queries:.2%} of them", file=sys.stderr)
print(f"{short_queries} skipped due to insufficient length (<{length_cutoff}) chars", file=sys.stderr)
print(f"\n===\nSherlock Stats\n===")
print(f"note -- sherlock is currently only run when atire detects the correct query")
print(f"Average similarity % between paraphrased query and original: {np.average(correct_percentages):.2f}%")
print(f"Average similarity % between paraphrased query and incorrect documents: {np.average(incorrect_avg_percentages):.2f}%")