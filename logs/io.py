from os import makedirs as mkdir, path as osp, stat

# try:
#     from iopath.common.file_io import g_pathmgr
# except ImportError:
#     from os import path as g_pathmgr


def makedirs(dir_path, *args, **kwargs):
    """
    Create the directory if it does not exist (wrapper).
    """
    # is_success = False
    # try:
    #     if not g_pathmgr.exists(dir_path):
    #         g_pathmgr.mkdirs(dir_path)
    #     is_success = True
    # except BaseException:
    #     print(f"Error creating directory: {dir_path}")
    exist_ok = kwargs.get('exist_ok', True)
    return mkdir(dir_path, exist_ok=exist_ok)  # is_success


def is_file_empty(file_path):
    """Check if file is empty by confirming if its size is 0 bytes"""
    # Check if file exist and it is empty
    return osp.exists(file_path) and stat(file_path).st_size == 0


__all__ = ['makedirs', 'is_file_empty']
