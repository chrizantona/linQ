import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deploy.vlm_http import local_endpoint
from deploy.vlm_server import parse_answer, prompt_for


class LocalVLMEndpointTests(unittest.TestCase):
    def test_only_loopback_infer_urls_are_accepted(self):
        for url in ('http://127.0.0.1:18775/infer', 'http://localhost:18775/infer', 'http://[::1]:18775/infer'):
            self.assertEqual(local_endpoint(url), url)
        for url in ('https://127.0.0.1:18775/infer', 'http://example.com/infer',
                    'http://127.0.0.1:18775/other', 'http://user@localhost:18775/infer',
                    'http://localhost:18775/infer?redirect=1'):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'local /infer'):
                local_endpoint(url)

    def test_vlm_http_url_validation_does_not_use_external_endpoints(self):
        from scripts.linq_prototype import environment
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'VLM_HTTP_URL': 'https://provider.invalid/infer'}):
            root = Path(directory)
            config = root / '.runtime/vlm_remote.json'
            config.parent.mkdir()
            config.write_text('{"url":"https://provider.invalid/infer"}')
            values = environment(root)
            self.assertEqual(values['VLM_HTTP_URL'], 'http://127.0.0.1:18775/infer')
            self.assertNotIn('VLM_CA_FILE', values)

    def test_local_prompt_and_parser_preserve_visible_evidence_protocol(self):
        prompt = prompt_for('05', 'A worker beneath an actually suspended load.')
        self.assertIn('Never infer unseen height, anchorage, permitted areas, load weight, movement, or intent.', prompt)
        self.assertIn('worker beneath an actually suspended load', prompt)
        self.assertIn('null', prompt)
        valid = '{"unsafe":null,"reason":"Context is unclear.","evidence":[],"scope":"05"}'
        parsed, error = parse_answer(valid, '05')
        self.assertIsNone(error)
        self.assertIsNone(parsed['unsafe'])
        invalid = '{"unsafe":"false","reason":"Visible.","evidence":[],"scope":"05"}'
        self.assertIsNone(parse_answer(invalid, '05')[0])
        self.assertIsNone(parse_answer(valid, '04')[0])


if __name__ == '__main__':
    unittest.main()
