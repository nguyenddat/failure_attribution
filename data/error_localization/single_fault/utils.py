def dataset_name_to_filename(name: str) -> str:
    return name.replace("/", "__").replace("&", "_and_")
