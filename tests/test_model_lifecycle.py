"""Check that sequential inference does not retain detector instances."""
from pathlib import Path
import sys
from threading import RLock
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference


class ModelLifecycleTest(unittest.TestCase):
    def test_detector_released_after_success_and_failure(self):
        references = []

        class Detector:
            pass

        def load(path):
            self.assertTrue(all(ref() is None for ref in references))
            detector = Detector()
            references.append(weakref.ref(detector))
            return detector, RLock()

        def predict(detector, lock, image, confidence, iou, size, device):
            if image == 'invalid':
                raise ValueError('test inference failure')
            return {'result': 'ok'}

        with patch.object(inference, 'load_detector', new=load), \
             patch.object(inference, 'predict', new=predict):
            for _ in range(6):
                self.assertEqual(inference.run_detector('model.pt', None, .25, .7, 640, 'cpu'),
                                 {'result': 'ok'})
            with self.assertRaises(ValueError):
                inference.run_detector('model.pt', 'invalid', .25, .7, 640, 'cpu')
            self.assertTrue(all(ref() is None for ref in references))


if __name__ == '__main__':
    unittest.main()
