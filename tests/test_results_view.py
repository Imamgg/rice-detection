"""Regression check for the non-empty detection/chart path on Python 3.14."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest


class ResultsViewTest(unittest.TestCase):
    def test_nonempty_detection_renders_chart_and_exports(self):
        self.check_results(1)

    def test_all_six_models_render_results(self):
        self.check_results(6)

    def check_results(self, model_count):
        image = Image.new('RGB', (64, 64), color='black')
        upload = io.BytesIO()
        image.save(upload, format='PNG')
        output = {
            'image': image,
            'rows': pd.DataFrame([['ciherang', .9, 1., 2., 20., 30.]],
                columns=['varietas', 'confidence', 'x1', 'y1', 'x2', 'y2']),
            'names': ['ciherang', 'ir64', 'mentik'],
            'elapsed_ms': 15.,
            'inference_ms': 10.,
        }
        app = Path(__file__).resolve().parents[1] / 'app.py'
        with patch('streamlit.file_uploader', return_value=upload), \
             patch('inference.run_detector', return_value=output) as runner:
            at = AppTest.from_file(str(app), default_timeout=60).run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            at.multiselect[0].set_value(at.multiselect[0].options[:model_count]).run()
            at.button[0].click().run()
            self.assertFalse(at.exception, [e.message for e in at.exception])
            self.assertEqual(at.metric[0].value, '1')
            self.assertEqual(runner.call_count, model_count)
            self.assertEqual(len(at.get('arrow_vega_lite_chart')), model_count)
            downloads = at.get('download_button')
            self.assertEqual(len(downloads), model_count + 1)
            self.assertEqual(len(at.dataframe[0].value), model_count)
            self.assertEqual(at.dataframe[0].value.iloc[0]['Butir'], 1)


if __name__ == '__main__':
    unittest.main()
