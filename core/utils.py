
import json

counter = 0


def save_data_as_json(data, path):
    with open(path, "w") as file:
        file.write(json.dumps(data, indent=4))


