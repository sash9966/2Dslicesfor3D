import re

def extract_filename(path):
    # Extract the number
    img_nr_match = re.search(r"\d{4}", path)
    if img_nr_match:
        img_nr = int(img_nr_match.group())
    else:
        raise ValueError("Could not find a 4-digit number in the filename.")

    # Extract the rotation identifier
    rotation_match = re.search(r"r[0-9]+", path)
    if rotation_match:
        rotation_str = rotation_match.group()
    else:
        rotation_str  = None
    
    return img_nr, rotation_str