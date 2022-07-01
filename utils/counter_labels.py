import json
 
val = "bdd100k_labels_images_val.json"
train = "bdd100k_labels_images_train.json"

clear = 0
rainy = 0
snowy = 0
overcast = 0
undefined = 0
partlyC = 0
foggy = 0

total = 0

with open(val) as json_file:
    data = json.load(json_file)

    for i in data:
        cmd = i["attributes"]["weather"]
        total += 1
        if cmd == "clear":
            clear += 1
        elif cmd == "foggy":
            foggy += 1
        elif cmd == "rainy":
            rainy += 1
        elif cmd == "snowy":
            snowy += 1
        elif cmd == "overcast":
            overcast += 1
        elif cmd == "undefined":
            undefined += 1    
        elif cmd == "partly cloudy":
            partlyC += 1

print("Il numero Clear e': ", clear)
print("Il numero Rainy e': ", rainy)
print("Il numero Overcast e': ", overcast)
print("Il numero Partly Cloud e': ", partlyC)
print("Il numero Foggy e': ", foggy)
print("Il numero Undefined e': ", undefined)
print("Il numero Snowy e': ", snowy)

print("Prova di somma: ", clear + rainy + snowy + overcast + undefined + partlyC + foggy)

print("Il numero totale e': ", total)
