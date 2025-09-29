"""
check the content of a pkl file or a directory of pkl files

"""
import pickle
import os

def inspect_pkl(file_path):
    if not os.path.exists(file_path):
        print(f"file not exit: {file_path}")
        return

    with open(file_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(f"Can not load pkl file : {e}")
            return

    # print the basic info
    print(f"\n📁 File path: {file_path}")
    print(f"🧩 Data type: {type(data)}")

    # if it is a list (e.g. multiple demos)
    if isinstance(data, list):
        print(f"🗂️  Data is a list, containing {len(data)} elements")
        for i, item in enumerate(data[:3]):  # print the first 3 demo examples
            print(f"  ┣━━ demo_{i} type: {type(item)}")
            if isinstance(item, dict):
                _inspect_dict(item, indent="  ┃ ")
            else:
                print(f"  ┃   ┗━ content: {item}")
        if len(data) > 3:
            print(f"  ┗━━ There are {len(data) - 3} more...")

    # if it is a dict (single demo)
    elif isinstance(data, dict):
        _inspect_dict(data)

    # other type
    else:
        print(f"📝 Content: {data}")


def _inspect_dict(d, indent=""):
    print(f"{indent}key-value pairs:")
    for k, v in d.items():
        if hasattr(v, 'shape'):  # NumPy array 或 Tensor
            print(f"{indent}├── {k}: {type(v).__name__}, shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"{indent}├── {k}: list, length={len(v)}")
        elif isinstance(v, dict):
            print(f"{indent}├── {k}: dict, keys={list(v.keys())}")
            _inspect_dict(v, indent + "   ")
        else:
            print(f"{indent}├── {k}: {type(v).__name__}")


if __name__ == "__main__":
    pkl_file = "demos.pkl"  # modify to your file path

    if os.path.isdir(pkl_file):
        # if it is a directory, traverse all .pkl files
        for root, _, files in os.walk(pkl_file):
            for fname in sorted(files):
                if fname.endswith('.pkl'):
                    inspect_pkl(os.path.join(root, fname))
    else:
        inspect_pkl(pkl_file)