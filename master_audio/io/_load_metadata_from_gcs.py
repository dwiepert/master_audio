#built-in
import io
import tempfile
from pathlib import Path
import json

#third party



def load_metadata_from_gcs(bucket, gcs_prefix, uid, extension = None):
    """
    load audio from google cloud storage
    :param bucket: gcs bucket object
    :param gcs_prefix: prefix leading to object in gcs bucket
    :param uid: audio identifier
    :param extension: data type (default, None)
    :return:  loaded metadata as a dictionary
    """
   
    if extension is None:
        extension = 'json'
        
    gcs_metadata_path = f'{gcs_prefix}/{uid}/metadata.{extension}'
    
    blob = bucket.blob(gcs_metadata_path)
    with tempfile.TemporaryDirectory() as tempdir:
            #temppath = Path(tempdir) /'metadata1.json'
        temppath = Path(tempdir) /'metadata.json'
        blob.download_to_filename(temppath)

        with open(temppath, 'r') as f:
            md = json.load(f)
            if isinstance(md, str):
                md = json.loads(md)
   
    return md