import unittest

from hinky.datasets import DatasetModule, HuggingFaceDatasetConfig, get_dataset


class TestBuildDatasets(unittest.TestCase):
    @unittest.skip("Requires network access and HuggingFace dataset download")
    def test_build_dataset(self):
        config = HuggingFaceDatasetConfig(hf_dataset_uri="cifar10")
        dataset_module = get_dataset(config)
        self.assertIsInstance(dataset_module, DatasetModule)
        for split in [dataset_module.train, dataset_module.val, dataset_module.test]:
            if split is None:
                continue
            batch = next(iter(split))
            self.assertIn("image", batch)


if __name__ == "__main__":
    unittest.main()
