from .huggingface import search_hf_datasets
from .kaggle import search_kaggle_datasets
from .paperswithcode import search_pwc_datasets
from .openml import search_openml_datasets
from .zenodo import search_zenodo_datasets

SOURCE_REGISTRY = {
    "Hugging Face": search_hf_datasets,
    "Kaggle": search_kaggle_datasets,
    "Papers with Code": search_pwc_datasets,
    "OpenML": search_openml_datasets,
    "Zenodo": search_zenodo_datasets,
}
