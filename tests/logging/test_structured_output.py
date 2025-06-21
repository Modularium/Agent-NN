import os
import json
import unittest
from core.logging_utils import init_logging
import io
import sys

class TestStructuredLogging(unittest.TestCase):
    def test_json_output(self):
        os.environ['LOG_FORMAT'] = 'json'
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            logger = init_logging('test_service')
            log = logger.bind(context_id='c1', session_id='s1', agent_id='a1')
            log.info('test_event', event='test')
        finally:
            sys.stdout = old
        output = buf.getvalue().strip()
        data = json.loads(output)
        self.assertEqual(data['service'], 'test_service')
        self.assertEqual(data['event'], 'test')
        self.assertEqual(data['context_id'], 'c1')

if __name__ == '__main__':
    unittest.main()
