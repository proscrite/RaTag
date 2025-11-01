#!/usr/bin/env python3
import os
import argparse

def fix_extensions(folder):
    count = 0
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            root, ext = os.path.splitext(fname)
            new_name = None

            if ext.lower() == "":
                # No extension → add .wfm
                new_name = fname + ".wfm"
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
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a directory")
        return

    fix_extensions(args.directory)

if __name__ == "__main__":
    main()
