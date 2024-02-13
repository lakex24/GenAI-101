import numpy as np


class BaseGenerator:
    def __init__(self):
        # self.config = config
        pass

    def generate(self):
        raise NotImplementedError

    def get_probability(self, x):
        raise NotImplementedError

class HistogramGenerator(BaseGenerator):

    GEN_SETTINGS = {
        'bins': 8,
        'density': True  # when you set True, total area under the histogram will sum to 1.0
        # when set to False, it will return the number of samples in each bin (discrete)
    }

    def __init__(self):
        super().__init__()  # Call the constructor of the BaseClass
        self.bin_probabilities = []
        self.bin_edges = []

    def build(self, data, in_settings=None):

        # Prepare settings
        settings = HistogramGenerator.GEN_SETTINGS.copy()
        if in_settings is not None:
            settings.update(in_settings)
        
        bins = settings['bins']
        density = settings['density']

        # when you set density=True, the total area under the histogram will sum to 1
        self.bin_probabilities, self.bin_edges = \
            np.histogram(data, bins=bins, density=density)


    def get_probability(self, x):
        
        assert len(self.bin_probabilities) == len(self.bin_edges) - 1

        # find the bin that x falls into
        find_prob = False
        prob = -1.0
        
        for i, _ in enumerate(self.bin_edges):

            if self.bin_edges[i] <= x < self.bin_edges[i + 1]:
                find_prob = True
                prob = self.bin_probabilities[i]
                break

        # corner case where 'x' is the last bin edge
        if x == self.bin_edges[-1]:
            find_prob = True
            prob = self.bin_probabilities[-1]

        if not find_prob:
            raise ValueError(f'x={x} is out of range')

        # return the probability of that bin
        return prob


    def generate(self):
        # print("hist. generator")
        # keep sampling until we get a sample thats is valid
        max_prob = max(self.bin_probabilities)
        

        while True:
            # sample in minimum and maximum range
            sample = np.random.uniform(self.bin_edges[0], self.bin_edges[-1])

            prob = self.get_probability(sample)

            # accept the sample with probability = prob / max_prob
            if np.random.uniform(0, max_prob) < prob:
                return sample

    