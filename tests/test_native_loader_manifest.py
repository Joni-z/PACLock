import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from paclock_bench.data.biot_dataset import build_biot_dataloaders
from paclock_bench.data.labram_dataset import build_labram_dataloaders


class NativeLoaderManifestTests(unittest.TestCase):
    def test_metadata_and_native_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(created_utc='2026-09-13T00:00:00+00:00',
                            dataset='synthetic', protocol={'fixture': True})
            (root/'manifest.json').write_text(json.dumps(manifest))
            signal = np.arange(64, dtype=np.float32).reshape(4,2,8)-20
            labels = np.array([0,1,1,0], dtype=np.int64)
            for split in ('train','val','test'):
                np.save(root/f'{split}_signals.npy', signal)
                np.save(root/f'{split}_labels.npy', labels)
            cfg = dict(data_root=tmp, batch_size=2, num_workers=0)
            for build in (build_labram_dataloaders, build_biot_dataloaders):
                tr, va, te, info = build(cfg)
                self.assertEqual(info['manifest'], manifest)
                self.assertEqual(info['class_counts']['val'], [2,2])
                self.assertEqual(info['n_samples']['val'], 4)
                for i in range(4):
                    x, y = va.dataset[i]
                    expected = signal[i]/100 if build is build_labram_dataloaders else (
                        signal[i]/(np.quantile(np.abs(signal[i]), q=.95,
                                              method='linear', axis=-1, keepdims=True)+1e-8))
                    np.testing.assert_array_equal(x.numpy(), expected)
                    self.assertEqual(y, int(labels[i]))

    def test_missing_manifest_fails_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            for build in (build_labram_dataloaders, build_biot_dataloaders):
                with self.assertRaisesRegex(FileNotFoundError, 'manifest.json'):
                    build(dict(data_root=tmp, num_workers=0))


if __name__ == '__main__':
    unittest.main()
