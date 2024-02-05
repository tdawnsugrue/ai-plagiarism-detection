import json
import numpy as np

def get_top_hits(results_file: str, prompts_file, top_n = 1):
    # WE DONT WANT TO OPEN THIS HERE... slow... zzz...
    

    search_results = open(results_file).read()
    hits = search_results[search_results.find("<hit>"):search_results.find("</hits>")].split()

    top_n = top_n if top_n <= len(hits) else -1

    ids = [prompts_file[retrieve_ids(h)[1]]['assistant'] for h in hits]
    return ids


def retrieve_ids(hit: str):
    id = hit[hit.find("<id>")+4:hit.find("</id>")]
    name = hit[hit.find("<name>")+6:hit.find("</name>")]
    # name is (probably) the one we want
    return (id, name)


if __name__ == "__main__":
    f = open("generated-flat.json")
    prompts = json.loads(f.read())[0]
    f.close()

    search_results = open("test-results.txt").read()

    query = search_results[search_results.find("<query>")+7:search_results.find("</query>")]

    hits = search_results[search_results.find("<hit>"):search_results.find("</hits>")].split()

    ids = retrieve_ids(hits[0])

    print(prompts[ids[0]], "\n=========\n")
    # this is (probably) the correct one
    print(prompts[ids[1]], "\n=========\n")
