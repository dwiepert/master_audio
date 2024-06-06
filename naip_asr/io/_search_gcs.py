def search_gcs(pattern: str, directory: str, bucket) -> list:
    """
    Search gcs bucket based on prefix. 
    :param pattern: str, pattern to search for
    :param directory: str, directory in gcs to search
    :param bucket: gcs bucket
    :return: list of files
    """
    files = []
    blobs = bucket.list_blobs(prefix=directory)
    for blob in blobs:
        name = blob.name
        if pattern in name:
            files.append(name)
    
    return files