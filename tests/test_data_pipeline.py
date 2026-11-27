import unittest

import numpy as np
from PIL import Image, ImageDraw

from space.src.preprocess import preprocess_image as preprocess_space_image
from src.data_loader import split_dataset
from src.preprocess import preprocess_image


class PreprocessingTests(unittest.TestCase):
    def test_dark_and_light_backgrounds_use_the_same_polarity(self):
        image = Image.new("L", (64, 64), 255)
        ImageDraw.Draw(image).line((14, 14, 50, 50), fill=0, width=5)
        inverted_image = Image.fromarray(255 - np.asarray(image))

        processed = preprocess_image(image)
        processed_inverted = preprocess_image(inverted_image)

        self.assertEqual(processed.shape, (64, 64, 1))
        self.assertEqual(processed.dtype, np.float32)
        np.testing.assert_array_equal(processed, processed_inverted)
        np.testing.assert_array_equal(processed, preprocess_space_image(image))


class DatasetSplitTests(unittest.TestCase):
    def test_reported_dataset_has_expected_stratified_split_sizes(self):
        labels = np.repeat(np.arange(10), 935)
        images = np.arange(len(labels))

        train, validation, test = split_dataset(images, labels)

        self.assertEqual(
            (len(train[0]), len(validation[0]), len(test[0])),
            (6544, 1403, 1403),
        )
        self.assertTrue(
            all(
                np.array_equal(np.unique(split[1]), np.arange(10))
                for split in (train, validation, test)
            )
        )


if __name__ == "__main__":
    unittest.main()
