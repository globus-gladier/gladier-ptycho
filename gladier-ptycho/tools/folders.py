def get_folder_paths(path):
    import glob
    import re
    
    return sorted(glob.glob(path, recursive=False),
                key = lambda v : int(re.search(r"(\d+)" , v[len(v)-"".join(reversed(v)).index('/'):]).group(0)))


def get_file_paths(path):
    import glob
    import re
    
    return sorted(glob.glob(path, recursive=False),
                key = lambda v : int(re.search(r"(\d+)" , v[len(v)-"".join((reversed(v))).index('/'):v.index('.')]).group(0)))