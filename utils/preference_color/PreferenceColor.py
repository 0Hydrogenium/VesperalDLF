import json


def init_preference_color():
    config_path = "./utils/preference_color/color_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg


class PreferenceColor:

    cfg = init_preference_color()

