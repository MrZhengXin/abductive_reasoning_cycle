import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
args = parser.parse_args()


lines = open(args.input_file).readlines()
random.shuffle(lines)
open(args.input_file, 'w').writelines(lines)