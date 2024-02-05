import json
import numpy as np



f = open("output-with-paraphrasing.json")
everything = json.loads(f.read())

llama_short = []
dipper_short = []

thres = 50

for i in range(len(everything)):
    if len(everything[i]['assistant']) < thres:
        llama_short.append(i)
    if len(everything[i]['paraphrased']) < thres:
        dipper_short.append(i)

both_short = np.intersect1d(llama_short, dipper_short)

print(len(everything))
print(f"{len(llama_short)}llama output indices with len < {thres}:")
#print(llama_short)
print(f"{len(dipper_short)}paraphrased entries with len < {thres}:")
#print(dipper_short)
print(f"{len(both_short)} entries shared between both groups")

problematic = []

for i in dipper_short:
    problematic.append(everything[i])

f = open("problematic-dipper-entries.json", "w")
json.dump(problematic, f)
f.close()

print("wrote problematic DIPPER entries to json")