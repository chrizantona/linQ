import unittest
from unittest.mock import patch

import numpy as np

from deploy.linq_tpu import detect, global_class_nms, map_tile_detections, merge_tiled_detections, tile_regions


class LinQTilingTests(unittest.TestCase):
    def test_regions_cover_source_and_match_documented_overlap(self):
        regions = tile_regions(1280, 720, 2, .2)
        self.assertEqual(regions, [[0, 0, 768, 432], [512, 0, 1280, 432], [0, 288, 768, 720], [512, 288, 1280, 720]])
        covered = np.zeros((720, 1280), dtype=bool)
        for x1, y1, x2, y2 in regions:
            covered[y1:y2, x1:x2] = True
        self.assertTrue(covered.all())
        self.assertEqual(regions[0][2] - regions[1][0], 256)

    def test_four_local_boxes_map_to_one_source_person(self):
        regions = tile_regions(1280, 720)
        source = [600., 300., 650., 400.]
        merged = []
        for index, region in enumerate(regions):
            x, y = region[:2]
            local = {'label': 'person', 'xyxy': [source[0] - x, source[1] - y, source[2] - x, source[3] - y], 'score': .9 - index * .1}
            mapped = map_tile_detections([local], region)
            self.assertEqual(mapped[0]['xyxy'], source)
            self.assertEqual(local['xyxy'][0], source[0] - x)
            merged.extend(mapped)
        kept = global_class_nms(merged)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['score'], .9)

    def test_nms_preserves_another_class_and_distinct_person(self):
        values = [{'label': 'person', 'xyxy': [0, 0, 100, 100], 'score': .9},
                  {'label': 'person', 'xyxy': [10, 0, 110, 100], 'score': .8},
                  {'label': 'person', 'xyxy': [150, 0, 180, 100], 'score': .7},
                  {'label': 'forklift', 'xyxy': [0, 0, 100, 100], 'score': .6}]
        kept = global_class_nms(values, .5)
        self.assertEqual([x['score'] for x in kept], [.9, .7, .6])

    def test_invalid_mapping_and_grid_fail(self):
        with self.assertRaises(ValueError):
            map_tile_detections([{'label': 'person', 'xyxy': [-1, 0, 10, 20], 'score': .9}], [0, 0, 100, 100])
        for grid, overlap in [(3, .2), (2, -1), (2, 1), (2, float('nan'))]:
            with self.assertRaises(ValueError):
                tile_regions(1280, 720, grid, overlap)

    def test_default_has_one_call_and_tiling_has_five_separate_receipts(self):
        detector = {'sha256': 'a' * 64, 'shape': [1, 3, 384, 384], 'decoder_sha256': 'b' * 64, 'compile_report_sha256': 'c' * 64}
        def result(name):
            return {'model': 'rfdetr', 'detections': [], 'receipt': {'status': 'received_and_validated', 'request_id': name}}
        with patch('deploy.linq_tpu.cached_person_detector', return_value=detector), patch('deploy.linq_tpu.detect_person', return_value=result('full')) as run:
            plain = detect(np.zeros((720, 1280, 3), dtype=np.uint8), {}, {})
            self.assertEqual(run.call_count, 1)
            self.assertNotIn('tiled_receipts', plain['provenance'])
        calls = [result(name) for name in ['full', 'tl', 'tr', 'bl', 'br']]
        with patch('deploy.linq_tpu.cached_person_detector', return_value=detector), patch('deploy.linq_tpu.detect_person', side_effect=calls) as run:
            tiled = detect(np.zeros((720, 1280, 3), dtype=np.uint8), {}, {'tiling': {'grid': 2, 'overlap': .2}})
            self.assertEqual(run.call_count, 5)
            self.assertEqual(tiled['provenance']['receipt']['request_id'], 'full')
            self.assertEqual([r['receipt']['request_id'] for r in tiled['provenance']['tiled_receipts']], ['tl', 'tr', 'bl', 'br'])
            self.assertTrue(tiled['provenance']['TPU_execution_verified'])

    def test_same_tile_duplicates_do_not_count_as_two_supports(self):
        box = {'label': 'person', 'xyxy': [10, 10, 30, 60], 'score': .9}
        merged, clusters = merge_tiled_detections([], [{'detections_source_xyxy': [box, {**box, 'score': .8}]}], 2)
        self.assertEqual(merged, [])
        self.assertEqual(clusters[0]['tile_support'], 1)

    def test_two_distinct_tiles_produce_one_supported_extra(self):
        large = {'label': 'person', 'xyxy': [658., 276., 687., 324.], 'score': .425}
        small = {'label': 'person', 'xyxy': [660., 277., 680., 309.], 'score': .29}
        merged, clusters = merge_tiled_detections([], [{'detections_source_xyxy': [large]}, {'detections_source_xyxy': [small]}], 2)
        self.assertEqual(merged, [large])
        self.assertEqual(clusters[0]['supporting_tile_indices'], [0, 1])

    def test_full_boxes_and_scores_survive_higher_scoring_tile_duplicates(self):
        full = [{'label': 'person', 'xyxy': [10, 10, 30, 60], 'score': .3}, {'label': 'person', 'xyxy': [12, 10, 32, 60], 'score': .28}]
        tile = {**full[0], 'score': .99}
        merged, clusters = merge_tiled_detections(full, [{'detections_source_xyxy': [tile]}, {'detections_source_xyxy': [tile]}], 2)
        self.assertEqual(merged, full)
        self.assertEqual(clusters[0]['reason'], 'already_present_full_frame')

    def test_invalid_support_count_is_rejected(self):
        for count in [0, 5, 2.0, True]:
            with self.assertRaises(ValueError):
                merge_tiled_detections([], [], count)


if __name__ == '__main__':
    unittest.main()
