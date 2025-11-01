#!/usr/bin/env python3
import os
import argparse

def fix_name_extensions(folder, pattern):
    count = 0
    
    for i, fname in enumerate(sorted(os.listdir(folder))):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            if pattern is not None:
                new_name = pattern + str(i).zfill(4) + '.wfm'
            else:
                root, ext = os.path.splitext(fname)
                new_name = fname
                if ext.lower() == "":
                    # No extension → add .wfm
                    new_name += ".wfm"
                elif ext.lower() == ".w":
                    # Replace .w with .wfm
                    new_name = root + ".wfm"

            if new_name:
                new_path = os.path.join(folder, new_name)
                os.rename(path, new_path)
                print(f"Renamed {fname} -> {new_name}")
                count += 1
    print(f"\nDone. Renamed {count} files.")

def main():
    parser = argparse.ArgumentParser(
        description="Fix file extensions: add .wfm if missing, or replace .w with .wfm"
    )
    parser.add_argument("directory", help="Path to the directory containing files")
    parser.add_argument("new_name", help="New name pattern for the files")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return

    fix_name_extensions(args.directory, args.new_name)

if __name__ == "__main__":
    main()
