# https://en.wikipedia.org/wiki/List_of_continents_and_continental_subregions_by_population

populations_by_continent = {
    'Asia': 59.4,
    'Africa': 17.6,
    'Europe': 9.4,
    'North America': 7.5,
    'South America': 5.5,
    'Oceania': 0.6
}

populations_by_continental_subregion = {
    'Southern Asia': 25.2,
    'Eastern Asia': 21.0,
    'South-eastern Asia': 8.5,
    'Eastern Africa': 5.8,
    'South America': 5.5,
    'Western Africa': 5.3,
    'Northern America': 4.7,
    'Eastern Europe': 3.7,
    'Western Asia': 3.7,
    'Northern Africa': 3.2,
    'Western Europe': 2.5,
    'Middle Africa': 2.4,
    'Central America': 2.2,
    'Southern Europe': 1.9,
    'Northern Europe': 1.3,
    'Central Asia': 1.0
    # exclude subregions (population ratio < 1%)
}

# GCP latency dashboard, 24-05-22, 6pm
region_l = ['asia-east1', 'africa-south1', 'australia-southeast1', 'europe-central2', 'me-central1', 'northamerica-northeast1', 'southamerica-east1', 'us-central2']
r2rlatency = {
    ('asia-east1', 'africa-south1'): 363,
    ('asia-east1', 'australia-southeast1'): 137,
    ('asia-east1', 'europe-central2'): 245,
    ('asia-east1', 'me-central1'): 139,
    ('asia-east1', 'northamerica-northeast1'): 179,
    ('asia-east1', 'southamerica-east1'): 285,
    ('asia-east1', 'us-central2'): 164,
    ('africa-south1', 'australia-southeast1'): 411,
    ('africa-south1', 'europe-central2'): 183,
    ('africa-south1', 'me-central1'): 252,
    ('africa-south1', 'northamerica-northeast1'): 217,
    ('africa-south1', 'southamerica-east1'): 329,
    ('africa-south1', 'us-central2'): 251,
    ('australia-southeast1', 'europe-central2'): 292,
    ('australia-southeast1', 'me-central1'): 188,
    ('australia-southeast1', 'northamerica-northeast1'): 198,
    ('australia-southeast1', 'southamerica-east1'): 294,
    ('australia-southeast1', 'us-central2'): 173,
    ('europe-central2', 'me-central1'): 135,
    ('europe-central2', 'northamerica-northeast1'): 101,
    ('europe-central2', 'southamerica-east1'): 210 ,
    ('europe-central2', 'us-central2'): 128,
    ('me-central1', 'northamerica-northeast1'): 195,
    ('me-central1', 'southamerica-east1'): 305,
    ('me-central1', 'us-central2'): 211,
    ('northamerica-northeast1', 'southamerica-east1'): 128,
    ('northamerica-northeast1', 'us-central2'): 29,
    ('southamerica-east1', 'us-central2'): 136
}

# intra-latency = 0.1ms


class Latency:
    def __init__(self, region):
        self.r2rlatency = r2rlatency
        self.region_list = region
        self.region_lowercase = []
        self.region2region()

    def region2region(self):
        for region in self.region_list:
            for re in region_l:
                if region.lowercase() in re:
                    self.region_lowercase.append(re)


class Task:
    def __init__(self, region_ratio):
        self.region_ratio = region_ratio
        self.region_list = list(region_ratio.keys())
