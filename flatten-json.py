# for flattening a 2d json into 1d (as necessitated by llama outputs)
import json



if __name__ == "__main__":

    f = open("generated-7b-chat.json")
    genchat = json.loads(f.read())
    f.close()

    f = open("generated-flat.json", "w")
    id = 1
    f.write("[{")
    for subset in genchat:
        for prompt in subset:
            if id != 1: f.write(",")
            f.write(f"\"{id}\" : {json.dumps(prompt)}\n")
            id += 1

    f.write("}]")
    f.close()
    
    print(f"flattened {id} prompts")
    