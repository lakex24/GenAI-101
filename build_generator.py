import logging
import argparse

import numpy as np

import read_csv_data as rcd
from generator import HistogramGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()

    # Load the input data (a distribution to be modeled and generated)
    csv_file_path = 'data/nba_players/all_seasons.csv'  # TODO: put it in args
    input_data = rcd.extract_nba_player_names_heights(csv_file_path)

    # We can build different types of generators here
    generator = build_generator_simple_histogram(input_data)

    # Now that we've built a generator, let's generate some samples
    sample_count = 1000

    samples = []
    for i in range(sample_count):
        sample = generator.generate()
        samples.append(sample)
    
    # Visualize the samples
    




def build_generator_simple_histogram(input_data: dict):
    """
    input_data: dict, key: player_name, value: player_height

    """
    logging.info("Building generator from input: %s", input)

    generator = HistogramGenerator()
    generator.build(data=list(input_data.values()))



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

