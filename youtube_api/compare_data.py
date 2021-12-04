import json

data = None
with open('data.json', 'r') as f:
    data = json.load(f)

if(data is not None):
    print(len(data))
    for key in data:
        with open("thumbnails/" + key + ".jpg", 'r') as f:
            if (f is None):
                print("image not found for associated data. KEY: " + key + " image: " + key + ".jpg" )
                exit()
    datafiltered = {x: y for x, y in data.items() if y['title'].isascii()}
    with open('datafiltered.json', 'w') as f:
        json.dump(datafiltered, f)
    
    print(len(data))
    print(len(datafiltered))
