import numpy as np
import sys
import math
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from reformat_for_atire import format_for_query

test_prompt = "Yes, beta blockers are sometimes prescribed to treat thyroid storm, which is a life-threatening complication of an overactive thyroid gland (hyperthyroidism).\n\nBeta blockers work by blocking the effects of the hormone epinephrine (adrenaline) on the body, which can help to reduce the symptoms of thyroid storm, such as:\n\n* Rapid heart rate\n* Palpitations\n* Shortness of breath\n* Chest pain\n* Anxiety\n* Confusion\n\nThyroid storm can occur when the thyroid gland produces too much thyroid hormone, leading to an excessive increase in metabolic rate, heart rate, and blood pressure. This can lead to serious complications, such as heart failure, arrhythmias, and stroke.\n\nBeta blockers can help to reduce the symptoms of thyroid storm by blocking the effects of epinephrine on the body, which can help to slow down the heart rate and blood pressure, and reduce the risk of complications. They are usually given intravenously in the hospital setting, and the dose may need to be adjusted based on the patient's response to the treatment.\n\nIt is important to note that beta blockers are not a cure for hyperthyroidism, and they do not treat the underlying cause of the condition. Treatment of hyperthyroidism usually involves reducing the production of thyroid hormone, either through medication or surgery. Beta blockers are used to manage the symptoms of hyperthyroidism until the underlying cause can be treated.\n\nIn summary, beta blockers are sometimes prescribed to treat thyroid storm, which is a life-threatening complication of an overactive thyroid gland. They work by blocking the effects of epinephrine on the body, which can help to reduce the symptoms of thyroid storm, such as rapid heart rate, palpitations, and shortness of breath. However, beta blockers are not a cure for hyperthyroidism, and they do not treat the underlying cause of the condition. Treatment of hyperthyroidism usually involves reducing the production of thyroid hormone, either through medication or surgery."



# assumes you have a list of terms and the original document
def kl_divergence(document_freq, document_length, collection_count, collection_length):
    px = document_freq / document_length
    qx = collection_count / collection_length
    return px * math.log(px / qx)

def rank_by_kl(document, collection: dict, collection_length, top_n = -1):
    top_terms = tf_idf_terms([document])
    for term in top_terms:
        pass
        # print(term)


def get_top_n_terms(content: str, top_n : int, return_str=False):
    try:
        top_terms = tf_idf_terms([content])

        if return_str:
            termstring = ""
            for t in top_terms[:top_n] : termstring += t + " "
            return termstring
        else:
            return top_terms[:top_n]
    except:
        print("An error occurred in the TFIdf vectorizer!", file=sys.stderr)
        return []

def tf_idf_terms(content: list):
    # try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(content)
        #print(X)
        feature_names = np.array(vectorizer.get_feature_names_out())

        X_arr = X.toarray()
        X_sorted = np.argsort(X_arr).flatten()[::-1]
        return feature_names[X_sorted]
    # except:
    #     print("An error occurred in the tf-idf vectoriser!", file = sys.stderr)
    #     exit()

def get_document_counts(document: list):
    vectoriser = CountVectorizer
    X = vectoriser.fit_transform(document)

def get_collection(corpus: dict):
    vect = CountVectorizer(strip_accents="ascii")
    X = vect.fit_transform(corpus)
    c_size = np.sum(X)
    print("collection size:", c_size)

def get_collection_atire(file):
    f = open(file)
    words = f.readlines()
    testcounts = 0
    collection = {}
    for line in words:
        l = line.strip().split(" ")
        collection[l[0]] = int(l[1])
        testcounts += int(l[1])
    
    return(collection, testcounts)
        


if __name__ == "__main__":
    # f = open("generated-flat.json")
    # gendata = json.loads(f.read())[0]
    # corpus = []
    # for key in gendata:
    #     corpus.append(gendata[key]['assistant'])
    # get_collection(corpus)
    collection_dict, collection_length = get_collection_atire("collection_dictionary.txt")
    #rank_by_kl(test_prompt, -1, -1)