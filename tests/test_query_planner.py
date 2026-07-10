import unittest

from PIL import Image, ImageDraw

from src.multimodal_captcha.query_planner import execute_query_plan, plan_prompt


class QueryPlannerTests(unittest.TestCase):
    def test_plan_prompt_extracts_color_only_query(self):
        plan = plan_prompt("请点击所有红色物体", supported_objects=["汽车", "吉他"])

        self.assertEqual(plan.mode, "color_only")
        self.assertEqual(plan.color, "红色")
        self.assertEqual(plan.objects, [])

    def test_plan_prompt_extracts_object_plus_color_query(self):
        plan = plan_prompt("请点击红色汽车", supported_objects=["汽车", "公交车", "吉他"])

        self.assertEqual(plan.mode, "object_plus_attributes")
        self.assertEqual(plan.color, "红色")
        self.assertEqual(plan.objects, ["汽车"])

    def test_plan_prompt_maps_super_category_to_supported_objects(self):
        plan = plan_prompt("请点击所有交通工具", supported_objects=["汽车", "公交车", "吉他", "大提琴"])

        self.assertEqual(plan.mode, "object_only")
        self.assertEqual(plan.objects, ["汽车", "公交车"])

    def test_execute_query_plan_handles_color_only(self):
        image = Image.new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([70, 10, 118, 54], fill=(230, 20, 30))
        plan = plan_prompt("请点击所有红色物体", supported_objects=["汽车"])

        result = execute_query_plan(image, plan)

        self.assertEqual(result["selected_cells"], [1])
        self.assertEqual(result["mode"], "color_only")

    def test_execute_query_plan_filters_object_candidates_by_color(self):
        image = Image.new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([70, 10, 118, 54], fill=(230, 20, 30))
        draw.rectangle([134, 10, 182, 54], fill=(30, 80, 230))
        plan = plan_prompt("请点击红色汽车", supported_objects=["汽车"])

        def fake_object_predictor(prompt):
            self.assertEqual(prompt, "请点击所有汽车")
            return [1, 2]

        result = execute_query_plan(image, plan, object_predictor=fake_object_predictor)

        self.assertEqual(result["selected_cells"], [1])
        self.assertEqual(result["object_candidate_cells"], [1, 2])

    def test_execute_query_plan_can_apply_position_without_object_model(self):
        image = Image.new("RGB", (192, 192), "white")
        plan = plan_prompt("请点击左边的物体", supported_objects=["汽车"])

        result = execute_query_plan(image, plan)

        self.assertEqual(result["selected_cells"], [0, 3, 6])
        self.assertEqual(result["mode"], "position_only")


if __name__ == "__main__":
    unittest.main()
