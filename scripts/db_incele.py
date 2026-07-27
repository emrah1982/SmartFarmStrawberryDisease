"""Veritabanını terminalden incelemek için yardımcı script.

Docker içinde veya dışında çalışır:
    python scripts/db_incele.py                     # tabloları ve satır sayılarını listeler
    python scripts/db_incele.py --tablo analizler   # satırları gösterir
    docker exec cilek-tespit python scripts/db_incele.py --tablo analizler
"""

import argparse
import sqlite3
from pathlib import Path


def veritabani_yolu(varsayilan='storage/kayitlar.db') -> Path:
    import os
    url = os.environ.get('DATABASE_URL', '')
    if url.startswith('sqlite:///'):
        return Path(url.replace('sqlite:///', ''))
    return Path(varsayilan)


def main():
    ap = argparse.ArgumentParser(description='SQLite veritabanını incele (salt okunur)')
    ap.add_argument('--db', type=str, default=None, help='Veritabanı dosyası')
    ap.add_argument('--tablo', type=str, default=None, help='Satırları gösterilecek tablo')
    ap.add_argument('--limit', type=int, default=20, help='Gösterilecek satır sayısı')
    args = ap.parse_args()

    yol = Path(args.db) if args.db else veritabani_yolu()
    if not yol.exists():
        print(f'❌ Veritabanı bulunamadı: {yol}')
        return 1

    db = sqlite3.connect(f'file:{yol}?mode=ro', uri=True)   # salt okunur aç
    db.row_factory = sqlite3.Row
    print(f'📂 {yol}  ({yol.stat().st_size / 1024:.1f} KB)\n')

    tablolar = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]

    if not args.tablo:
        print('TABLOLAR')
        for t in tablolar:
            adet = db.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            sutunlar = [r['name'] for r in db.execute(f'PRAGMA table_info("{t}")')]
            print(f'  {t:<22} {adet:>6} satır   {", ".join(sutunlar)}')
        print('\nSatırları görmek için: --tablo <ad>')
        return 0

    if args.tablo not in tablolar:
        print(f'❌ Tablo yok: {args.tablo}\n   Mevcut: {", ".join(tablolar)}')
        return 1

    satirlar = db.execute(
        f'SELECT * FROM "{args.tablo}" ORDER BY rowid DESC LIMIT ?', (args.limit,)).fetchall()
    if not satirlar:
        print('(tablo boş)')
        return 0

    for s in satirlar:
        print('─' * 60)
        for k in s.keys():
            deger = s[k]
            if isinstance(deger, str) and len(deger) > 70:
                deger = deger[:67] + '...'
            print(f'  {k:<18} {deger}')
    print('─' * 60)
    toplam = db.execute(f'SELECT COUNT(*) FROM "{args.tablo}"').fetchone()[0]
    print(f'{len(satirlar)} / {toplam} satır gösterildi')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
