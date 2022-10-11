import h5py
from h5py import File


def print_attrs(name, obj):
    # Create indent
    shift = name.count('/') * '    '
    item_name = name.split("/")[-1]
    print(shift + item_name)
    try:
        for key, val in obj.attrs.items():
            print(shift + '    ' + f"{key}: {val}")
    except:
        pass

def check_h5_tree(fpath):
    f = File(fpath,'r')
    print(f.visititems(print_attrs))
    f.close()