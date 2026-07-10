import unittest

from PIL import Image, ImageDraw

from src.multimodal_captcha.attribute_scorer import (
    score_cell_colors,
    score_cell_positions,
    score_cell_sizes,
    select_cells_from_scores,
)


class AttributeScorerTests(unittest.TestCase):
    def test_score_cell_colors_detects_red_cells_without_training(self):
        image = Image.new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([70, 10, 118, 54], fill=(230, 20, 30))
        draw.rectangle([134, 10, 182, 54], fill=(30, 60, 220))

        scores = score_cell_colors(image, "红色")

        self.assertGreater(scores[1], 0.70)
        self.assertLess(scores[2], 0.25)
        self.assertEqual(select_cells_from_scores(scores, threshold=0.35), [1])

    def test_score_cell_colors_detects_blue_cells(self):
        image = Image.new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([134, 10, 182, 54], fill=(30, 80, 230))

        scores = score_cell_colors(image, "蓝色")

        self.assertEqual(select_cells_from_scores(scores, threshold=0.35), [2])

    def test_score_cell_positions_selects_left_column(self):
        scores = score_cell_positions("左边")

        self.assertEqual(select_cells_from_scores(scores, threshold=0.9), [0, 3, 6])

    def test_score_cell_sizes_prefers_larger_foreground_object(self):
        image = Image.new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([80, 20, 105, 45], fill=(30, 150, 60))
        draw.rectangle([135, 5, 188, 58], fill=(30, 150, 60))

        scores = score_cell_sizes(image)

        self.assertGreater(scores[2], scores[1])


if __name__ == "__main__":
    unittest.main()
