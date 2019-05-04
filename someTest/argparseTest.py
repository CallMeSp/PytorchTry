import argparse

parser = argparse.ArgumentParser(description='Train a Dialog QA model.')
parser.add_argument('-a', '--abc', help='a sourcedir', action='store_false')
parser.add_argument('-x', '--xyz', dest='mn', help='a sourcedir', nargs='?')
args = parser.parse_args()
print(args.abc)
print(args.mn)
