def upload_to_gcs(gcs_prefix, path, bucket):
    '''
    Upload a file to a google cloud storage bucket
    Inputs:
    :param gcs_prefix: path in the bucket to save file to (no gs://project-name in the path)
    :param path: local string path of the file to upload
    :param bucket: initialized GCS bucket object
    '''
    assert bucket is not None, 'no bucket given for uploading'
    blob = bucket.blob(str(gcs_prefix))
    blob.upload_from_filename(str(path))