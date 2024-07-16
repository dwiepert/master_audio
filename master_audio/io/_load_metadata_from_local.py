import json 

def load_metadata_from_local(input_dir, uid, extension = None):
    """
    :param input_directory: directory where data is stored locally
    :param uid: audio identifier
    :param extension: data type (default, None)
    :return: loaded metadata as a dictionary
    """
    
    if extension is None:
        extension = 'json'
        
    metadata_path = f'{input_dir}/{uid}/metadata.{extension}'
    with open(metadata_path, 'r') as f:
            md = json.load(f)
            if isinstance(md, str):
                md = json.loads(md)
    
    
    return md