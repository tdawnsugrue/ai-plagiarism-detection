import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from reformat_for_atire import format_for_query

test_prompt = ["Yes, beta blockers are sometimes prescribed to treat thyroid storm, which is a life-threatening complication of an overactive thyroid gland (hyperthyroidism).\n\nBeta blockers work by blocking the effects of the hormone epinephrine (adrenaline) on the body, which can help to reduce the symptoms of thyroid storm, such as:\n\n* Rapid heart rate\n* Palpitations\n* Shortness of breath\n* Chest pain\n* Anxiety\n* Confusion\n\nThyroid storm can occur when the thyroid gland produces too much thyroid hormone, leading to an excessive increase in metabolic rate, heart rate, and blood pressure. This can lead to serious complications, such as heart failure, arrhythmias, and stroke.\n\nBeta blockers can help to reduce the symptoms of thyroid storm by blocking the effects of epinephrine on the body, which can help to slow down the heart rate and blood pressure, and reduce the risk of complications. They are usually given intravenously in the hospital setting, and the dose may need to be adjusted based on the patient's response to the treatment.\n\nIt is important to note that beta blockers are not a cure for hyperthyroidism, and they do not treat the underlying cause of the condition. Treatment of hyperthyroidism usually involves reducing the production of thyroid hormone, either through medication or surgery. Beta blockers are used to manage the symptoms of hyperthyroidism until the underlying cause can be treated.\n\nIn summary, beta blockers are sometimes prescribed to treat thyroid storm, which is a life-threatening complication of an overactive thyroid gland. They work by blocking the effects of epinephrine on the body, which can help to reduce the symptoms of thyroid storm, such as rapid heart rate, palpitations, and shortness of breath. However, beta blockers are not a cure for hyperthyroidism, and they do not treat the underlying cause of the condition. Treatment of hyperthyroidism usually involves reducing the production of thyroid hormone, either through medication or surgery."]

# convert both features & col/freq to np arrays
# call an argsort(?refresh memory on this) to sort by frequency
# use this to get top x terms (or terms over a certain tf-idf threshold)
# throw into a query format and run thru atire as usual
# cross fingers and hope for the best

# .... we probably want to do some kind of reporting on what we're doing as we go
# output-with-paraphrasing contains both the original and paraphrased responses.
# probably include a <query terms -->>

def get_top_n_terms(content: str, top_n : int, return_str=False):
    content = [content]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(content)

        feature_names = np.array(vectorizer.get_feature_names_out())

        X_arr = X.toarray()
        X_sorted = np.argsort(X_arr).flatten()[::-1]
        if return_str:
            termstring = ""
            for t in feature_names[X_sorted][:top_n] : termstring += t + " "
            return termstring
        else:
            return feature_names[X_sorted][:top_n]
    except:
        print("An error occurred in the TFIdf vectorizer!", file=sys.stderr)
        return []