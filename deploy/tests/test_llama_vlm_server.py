import base64
import io
import json
import os
import hashlib
import shutil
import subprocess
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from deploy.llama_vlm_server import DEFAULT_MODEL, LlamaEngine, encode_content, loopback_endpoint, native_video_content
from deploy.vlm_server import Settings, create_app, prompt_for


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, maximum=-1):
        return self.payload if maximum < 0 else self.payload[:maximum]


class LlamaBridgeTests(unittest.TestCase):
    def test_external_backend_is_rejected(self):
        for url in ['https://example.com/v1/chat/completions', 'http://example.com/v1/chat/completions', 'http://127.0.0.1:18766/other']:
            with self.assertRaises(ValueError):
                loopback_endpoint(url)

    def test_three_images_preserve_order_and_actual_completion_is_not_rewritten(self):
        requests = []
        caption = '{"unsafe":null,"reason":"Visible people.","evidence":[],"scope":"open"}'
        def backend(request, **options):
            if isinstance(request, str):
                return Response({'status': 'ok'})
            requests.append(json.loads(request.data))
            return Response({'id': 'technical-test', 'model': DEFAULT_MODEL, 'choices': [{'message': {'content': caption}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 18}, 'timings': {'prompt_ms': 123.4, 'predicted_ms': 56.7, 'cache_n': 0}})
        with patch('deploy.llama_vlm_server.urllib.request.urlopen', side_effect=backend):
            engine = LlamaEngine(Settings(model=DEFAULT_MODEL, device='cpu', max_new_tokens=192))
            raw, execution = engine.generate([Image.new('RGB', (1200, 600), color) for color in ['red', 'green', 'blue']], prompt_for('open', ''))
        self.assertEqual(raw, caption)
        self.assertTrue(execution['actual_execution'])
        self.assertFalse(execution['TPU_execution_verified'])
        self.assertFalse(execution['GPU_execution_claimed'])
        self.assertEqual(requests[0]['max_tokens'], 192)
        constraints = requests[0]['response_format']
        self.assertEqual(constraints['type'], 'json_schema')
        self.assertIs(constraints['json_schema']['strict'], True)
        schema = constraints['json_schema']['schema']
        self.assertEqual(set(schema['required']), {'unsafe', 'reason', 'evidence', 'scope'})
        self.assertIs(schema['additionalProperties'], False)
        self.assertEqual(schema['properties']['unsafe']['type'], ['boolean', 'null'])
        self.assertEqual(schema['properties']['reason']['maxLength'], 160)
        self.assertEqual(schema['properties']['evidence']['maxItems'], 2)
        self.assertEqual(schema['properties']['scope']['type'], 'string')
        self.assertEqual(execution['response_constraints'], constraints)
        self.assertEqual(execution['max_new_tokens'], 192)
        self.assertEqual(execution['llama_timings'], {'prompt_ms': 123.4, 'predicted_ms': 56.7, 'cache_n': 0})
        self.assertEqual(execution['finish_reason'], 'stop')
        self.assertEqual(requests[0]['temperature'], 0)
        self.assertIs(requests[0]['cache_prompt'], True)
        self.assertIs(execution['cache_prompt'], True)
        self.assertFalse(requests[0]['chat_template_kwargs']['enable_thinking'])
        encoded = [item['image_url']['url'] for item in requests[0]['messages'][0]['content'] if item['type'] == 'image_url']
        self.assertEqual(len(encoded), 3)
        for data in encoded:
            image = Image.open(io.BytesIO(base64.b64decode(data.split(',', 1)[1])))
            self.assertLessEqual(max(image.size), 512)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for the real lossless codec contract')
    def test_native_video_preserves_two_and_three_post_jpeg_rgb_frames_in_order(self):
        for count in [2, 3]:
            with self.subTest(count=count):
                images = [Image.new('RGB', (32, 24), color) for color in ['red', 'green', 'blue'][:count]]
                prompt = prompt_for('open', 'technical fixture')
                reference, _ = encode_content(images, prompt)
                expected = []
                for part in reference:
                    if part['type'] == 'image_url':
                        data = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                        expected.append(Image.open(io.BytesIO(data)).convert('RGB').tobytes())
                requests = []
                answer = '{"unsafe":null,"reason":"Technical fixture","evidence":[],"scope":"open"}'
                def backend(request, **options):
                    if isinstance(request, str):
                        return Response({'status': 'ok'})
                    requests.append(json.loads(request.data))
                    return Response({'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}], 'timings': {'prompt_n': 555, 'prompt_ms': 12.5}, 'usage': {'completion_tokens': 8}})
                with patch('deploy.llama_vlm_server.urllib.request.urlopen', side_effect=backend):
                    engine = LlamaEngine(Settings(model=DEFAULT_MODEL, device='cpu', max_new_tokens=192), input_mode='native_video')
                    raw, execution = engine.generate(images, prompt)
                content = requests[0]['messages'][0]['content']
                videos = [part for part in content if part['type'] == 'input_video']
                self.assertEqual(len(videos), 1)
                self.assertFalse(any(part['type'] == 'image_url' for part in content))
                encoded = base64.b64decode(videos[0]['input_video']['data'])
                decoded = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', 'pipe:0', '-vf', 'fps=1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], input=encoded, check=True, capture_output=True).stdout
                self.assertEqual(decoded, b''.join(expected))
                self.assertEqual(execution['post_jpeg_rgb_sha256'], [hashlib.sha256(value).hexdigest() for value in expected])
                self.assertEqual(execution['ordered_frame_count'], count)
                self.assertEqual(execution['native_video_frame_timestamps_s'], list(range(count)))
                self.assertEqual(execution['input_representation'], 'native_video_ffv1_mkv_post_jpeg_rgb')
                self.assertEqual(content[-1]['text'], prompt)
                self.assertEqual(raw, answer)
                self.assertEqual(execution['llama_timings']['prompt_n'], 555)
                self.assertEqual(execution['finish_reason'], 'stop')
                self.assertGreaterEqual(execution['preparation_ms'], execution['video_encoding_ms'])
                self.assertEqual(requests[0]['temperature'], 0)
                self.assertFalse(requests[0]['chat_template_kwargs']['enable_thinking'])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for deterministic lossless video')
    def test_same_post_jpeg_inputs_have_stable_video_bytes_for_kv_cache(self):
        content, sizes = encode_content([Image.new('RGB', (32, 24), color) for color in ['red', 'blue']], 'technical fixture')
        first, first_metadata = native_video_content(content, sizes, 'technical fixture')
        second, second_metadata = native_video_content(content, sizes, 'technical fixture')
        self.assertEqual(first, second)
        self.assertEqual(first_metadata['native_video_sha256'], second_metadata['native_video_sha256'])
        self.assertEqual(first_metadata['post_jpeg_rgb_sha256'], second_metadata['post_jpeg_rgb_sha256'])

    def test_native_mode_single_frame_uses_image_without_running_codec(self):
        requests = []
        def backend(request, **options):
            if isinstance(request, str):
                return Response({'status': 'ok'})
            requests.append(json.loads(request.data))
            return Response({'choices': [{'message': {'content': 'actual completion'}, 'finish_reason': 'stop'}]})
        with patch('deploy.llama_vlm_server.urllib.request.urlopen', side_effect=backend), patch('deploy.llama_vlm_server.subprocess.run') as codec:
            engine = LlamaEngine(Settings(device='cpu', max_new_tokens=128), input_mode='native_video')
            raw, execution = engine.generate([Image.new('RGB', (20, 10))], 'technical fixture')
        codec.assert_not_called()
        self.assertEqual(raw, 'actual completion')
        self.assertEqual(execution['input_representation'], 'single_jpeg_image')
        self.assertTrue(any(part['type'] == 'image_url' for part in requests[0]['messages'][0]['content']))

    def test_native_mode_rejects_wrong_count_or_dimensions_before_backend_request(self):
        with patch('deploy.llama_vlm_server.urllib.request.urlopen', return_value=Response({'status': 'ok'})) as backend:
            engine = LlamaEngine(Settings(device='cpu', max_new_tokens=128), input_mode='native_video')
            for images in [[], [Image.new('RGB', (20, 20))]*4, [Image.new('RGB', (20, 20)), Image.new('RGB', (20, 10))]]:
                with self.assertRaises(ValueError):
                    engine.generate(images, 'technical fixture')
        self.assertEqual(backend.call_count, 1)

    def test_native_codec_failure_does_not_send_images_as_hidden_fallback(self):
        with patch('deploy.llama_vlm_server.urllib.request.urlopen', return_value=Response({'status': 'ok'})) as backend, patch('deploy.llama_vlm_server.subprocess.run', side_effect=FileNotFoundError('technical fixture')):
            engine = LlamaEngine(Settings(device='cpu', max_new_tokens=128), input_mode='native_video')
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                engine.generate([Image.new('RGB', (20, 20))]*2, 'technical fixture')
        self.assertEqual(backend.call_count, 1)

    def test_generation_budget_rejects_more_than_192_tokens(self):
        with self.assertRaises(ValueError):
            LlamaEngine(Settings(model=DEFAULT_MODEL, device='cpu', max_new_tokens=193))

    def test_invalid_model_json_is_returned_as_strict_protocol_error(self):
        def backend(request, **options):
            if isinstance(request, str):
                return Response({'status': 'ok'})
            return Response({'choices': [{'message': {'content': 'not JSON'}, 'finish_reason': 'stop'}]})
        buffer = io.BytesIO()
        Image.new('RGB', (24, 16)).save(buffer, format='PNG')
        payload = {'protocol': 'linq_vlm_frames_v1', 'scope': 'open', 'frames': [{'mime_type': 'image/png', 'base64': base64.b64encode(buffer.getvalue()).decode()}]}
        with patch.dict(os.environ, {'VLM_HTTP_TOKEN': 'test-only'}), patch('deploy.llama_vlm_server.urllib.request.urlopen', side_effect=backend):
            settings = Settings(model=DEFAULT_MODEL, device='cpu', max_new_tokens=128, audit_log=None)
            with TestClient(create_app(settings, engine_factory=LlamaEngine)) as client:
                response = client.post('/infer', json=payload, headers={'Authorization': 'Bearer test-only'})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['detail']['raw_answer'], 'not JSON')
                self.assertEqual(client.get('/health', headers={'Authorization': 'Bearer test-only'}).json()['actual_completed_generations'], 1)


if __name__ == '__main__':
    unittest.main()
