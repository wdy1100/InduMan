
"""
recursively print the H5 file structure
"""
import h5py
import os

def parse_h5_file(file_path):
    # check if file exists
    if not os.path.exists(file_path):
        print(f"Error: file '{file_path}' does not exist.")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\nParsing H5 file: {file_path}")
            print("="*60)

            # recursive function to print structure
            def print_structure(group, prefix=""):
                for key in group.keys():
                    item = group[key]
                    if isinstance(item, h5py.Dataset):
                        print(f"{prefix}Dataset: {key} -> shape: {item.shape}, type: {item.dtype}")
                    elif isinstance(item, h5py.Group):
                        print(f"{prefix}Group: {key}")
                    # print attributes (if any)
                    if item.attrs:
                        for attr_name, attr_value in item.attrs.items():
                            print(f"{prefix}  Attribute: {key}.{attr_name} = {attr_value}")
                    # recursive call for subgroup
                    if isinstance(item, h5py.Group):
                        print_structure(item, prefix + "  ")

            # print structure of h5 file
            print_structure(f)

    except Exception as e:
        print(f"Error parsing H5 file: {e}")

# main function
if __name__ == "__main__":
    path = "h5file.h5"
    parse_h5_file(path)