import base64
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from deploy.vlm_server import Settings, create_app, decode_request, encode_images, prompt_for


def payload(scope='open'):
    stream = io.BytesIO()
    Image.new('RGB', (24, 16), 'white').save(stream, format='PNG')
    return dict(protocol='linq_vlm_frames_v1', scope=scope, prompt='Inspect visible risk.', context={},
                frames=[dict(mime_type='image/png', base64=base64.b64encode(stream.getvalue()).decode())])


class FakeEngine:
    def __init__(self, settings):
        self.metadata = dict(model='mock', device='mock', execution_mode='mock_generation_test_only', actual_execution=False)
        self.raw = json.dumps(dict(scope='open', unsafe=None, reason='', evidence=[]))
        self.calls = []

    def generate(self, images, prompt):
        self.calls.append((images, prompt))
        return self.raw, dict(generated_tokens=12, actual_execution=False)


class VLMServerTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'VLM_HTTP_TOKEN': 'test-only-token'})
        self.environment.start()
        self.engine = FakeEngine(Settings())
        self.headers = {'Authorization': 'Bearer test-only-token'}
        self.settings = Settings(audit_log=None)

    def tearDown(self):
        self.environment.stop()

    def client(self, settings=None):
        return TestClient(create_app(settings or self.settings, engine_factory=lambda _: self.engine))

    def test_token_required_to_construct_service(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'VLM_HTTP_TOKEN'):
                create_app(self.settings, engine_factory=lambda _: self.engine)

    def test_auth_and_readiness(self):
        with self.client() as client:
            self.assertEqual(client.get('/health').status_code, 401)
            response = client.get('/health', headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['ready'])
            self.assertFalse(response.json()['execution']['actual_execution'])
            self.assertEqual(client.post('/infer', json=payload()).status_code, 401)

    def test_open_scope_null_result_and_mock_metadata(self):
        with self.client() as client:
            response = client.post('/infer', json=payload(), headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertIsNone(result['unsafe'])
            self.assertEqual(result['reason'], '')
            self.assertEqual(result['scope'], 'open')
            self.assertFalse(result['execution']['actual_execution'])
            self.assertEqual(len(self.engine.calls), 1)
            self.assertEqual(client.get('/health', headers=self.headers).json()['successful_generations'], 1)

    def test_numeric_scope_and_root_endpoint(self):
        self.engine.raw = json.dumps(dict(scope='05', unsafe=True, reason='Visible suspended load.', evidence=['Worker beneath load']))
        with self.client() as client:
            response = client.post('/', json=payload('05'), headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIs(response.json()['unsafe'], True)

    def test_invalid_output_raw_recorded_no_fabrication(self):
        self.engine.raw = 'not JSON'
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'requests.jsonl'
            with self.client(Settings(audit_log=str(log))) as client:
                response = client.post('/infer', json=payload(), headers=self.headers)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['detail']['raw_answer'], 'not JSON')
            recorded = json.loads(log.read_text())
            self.assertEqual(recorded['status'], 'invalid_model_output')
            self.assertEqual(recorded['raw_answer'], 'not JSON')
            self.assertNotIn('Bearer', log.read_text())

    def test_scope_mismatch_and_string_boolean_rejected(self):
        with self.client() as client:
            for output in [dict(scope='05', unsafe=False, reason='', evidence=[]),
                           dict(scope='open', unsafe='false', reason='', evidence=[])]:
                self.engine.raw = json.dumps(output)
                response = client.post('/infer', json=payload(), headers=self.headers)
                self.assertEqual(response.status_code, 422)

    def test_invalid_frames_and_limits_rejected_before_generation(self):
        with self.client(Settings(audit_log=None, max_frames=1, max_image_pixels=100)) as client:
            for body in [payload(), dict(payload(), frames=[]), dict(payload(), scope='unknown')]:
                self.assertEqual(client.post('/infer', json=body, headers=self.headers).status_code, 422)
            malformed = payload()
            malformed['frames'][0]['base64'] = '!invalid'
            self.assertEqual(client.post('/infer', json=malformed, headers=self.headers).status_code, 422)
            self.assertFalse(self.engine.calls)

    def test_body_limit(self):
        with self.client(Settings(audit_log=None, max_body_bytes=10)) as client:
            self.assertEqual(client.post('/infer', json=payload(), headers=self.headers).status_code, 413)
            self.assertFalse(self.engine.calls)

    def test_mime_mismatch_and_no_external_frame_urls(self):
        body = payload()
        body['frames'][0]['mime_type'] = 'image/jpeg'
        with self.assertRaisesRegex(ValueError, 'MIME'):
            decode_request(body, self.settings)
        body['frames'] = [dict(mime_type='image/png', url='https://example.invalid/image.png')]
        with self.assertRaises(ValueError):
            decode_request(body, self.settings)

    def test_prompt_has_visible_only_contract_for_both_scopes(self):
        for scope in ['open', '05']:
            prompt = prompt_for(scope, '')
            self.assertIn('null', prompt)
            self.assertIn('scope', prompt)
            self.assertIn(scope, prompt)


if __name__ == '__main__':
    unittest.main()
