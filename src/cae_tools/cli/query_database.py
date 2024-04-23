from cae_tools.utils.model_database import ModelDatabase
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database_path")
    args = parser.parse_args()
    md = ModelDatabase(args.database_path)
    md.dump()

if __name__ == '__main__':
    main()