import zipfile
file_name = "/workspace/open.zip"
output_dir = "/workspace/open"
zip_file = zipfile.ZipFile(file_name)
zip_file.extractall(path=output_dir)