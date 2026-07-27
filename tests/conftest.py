"""Tüm testler için ortak kurulum.

NEDEN BURADA?
    Depolama/veritabanı yönlendirmesi `app.main` İÇE AKTARILMADAN ÖNCE
    yapılmalı: motor ve /media bağlaması modül yüklenirken kurulur, sonradan
    değiştirilemez.

    Bu blok tek bir test dosyasında dursaydı, o dosya olmadan çalıştırılan
    testler (örn. `pytest tests/test_canli.py`) GERÇEK veritabanına yazardı.
    conftest.py her koşuda ve her dosyadan önce yüklendiği için tek güvenli
    yer burasıdır.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_tmp = tempfile.mkdtemp(prefix='cilek_test_')
os.environ['STORAGE_DIR'] = _tmp
os.environ['DATABASE_URL'] = f'sqlite:///{Path(_tmp) / "test.db"}'
