from pathlib import Path
import unittest

from scripts import prefetch_qwen_gguf


class PrefetchDefaultRootTests(unittest.TestCase):
    def test_default_root_is_this_repository_not_legacy_checkout(self):
        self.assertEqual(prefetch_qwen_gguf.DEFAULT_ROOT, Path(prefetch_qwen_gguf.__file__).resolve().parents[1])
        self.assertNotIn('linq_safety_proto', str(prefetch_qwen_gguf.DEFAULT_ROOT))


if __name__ == '__main__':
    unittest.main()
