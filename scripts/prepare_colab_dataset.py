"""Colab'de dataset'i Drive'dan alıp yerel diske hazırlar.

NEDEN AYRI SCRIPT?
    Notebook hücrelerindeki kod, Colab sekmesi açıkken önbellekte kalır;
    dosyayı güncellesek bile açık oturum eski kodu çalıştırır. Bu script
    depoda durduğu için notebook'un "git pull" adımı her çalıştırmada en
    güncel halini indirir — düzeltmeler anında yansır, notebook'u yeniden
    indirmeye gerek kalmaz.

NEDEN ÖNCE KOPYALA SONRA AÇ?
    Arşivi doğrudan Drive üzerinden açmak yavaştır: Drive FUSE dosya sistemi
    rastgele erişimde (zip merkezi dizini + 22.000 küçük dosya) yüksek gecikme
    üretir. Tek büyük dosyayı önce yerel diske SIRALI okumak, sonra yerelde
    açmak belirgin şekilde hızlıdır. Açma işi de mümkünse sistem 'unzip'
    komutuna verilir (C ile yazılmış, çok sayıda küçük dosyada Python'dan hızlı).

Usage (Colab):
    python scripts/prepare_colab_dataset.py \
        --drive-root /content/drive/MyDrive/SmartFarmStrawberryDisease \
        --repo /content/SmartFarmStrawberryDisease
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

ARCHIVE_NAMES = ('dataset_colab.zip', 'dataset.zip', 'dataset_colab.tar', 'dataset.tar')


def is_source(name: str) -> bool:
    """Birleşik dataset'in kaynak klasörleri."""
    return name == 'augmented_train' or name.endswith('.yolo26')


def sources_in(d: Path) -> List[Path]:
    try:
        return [q for q in d.iterdir() if q.is_dir() and is_source(q.name)]
    except OSError:
        return []


def find_archive(drive_root: Path, mydrive: Path) -> Optional[Path]:
    """Arşivi bilinen konumlarda, sonra MyDrive'da (4 seviyeye kadar) arar."""
    candidates = []
    for name in ARCHIVE_NAMES:
        candidates += [drive_root / 'dataset' / name, drive_root / name,
                       mydrive / name, mydrive / 'Colab Notebooks' / name]
    for c in candidates:
        if c.exists():
            return c

    if not mydrive.exists():
        return None
    for cur, dirs, files in os.walk(mydrive):
        if len(Path(cur).relative_to(mydrive).parts) >= 4:
            dirs[:] = []          # çok derine inip Drive'ı yavaşlatma
            continue
        for name in ARCHIVE_NAMES:
            if name in files:
                cand = Path(cur) / name
                try:
                    cand.stat()
                    return cand
                except OSError:
                    continue      # Drive kısayolu / erişilemeyen girdi
    return None


def archive_root(names: List[str]) -> Optional[str]:
    """Arşiv içindeki dataset kökünü İÇERİKTEN bulur (kök adı ne olursa olsun).

    Arşiv 'dataset/', 'dataset_colab/' köküyle veya sarmalayıcısız olabilir.
    Kaynak klasörlerin BİR ÜSTÜ dataset köküdür; yanlış seçilirse
    dataset/dataset_colab/... gibi fazladan seviye oluşur ve eğitim
    "images not found" verir.
    """
    for n in names:
        parts = n.split('/')
        for i, part in enumerate(parts[:-1]):
            if is_source(part):
                return '/'.join(parts[:i])     # '' = sarmalayıcı yok
    return None


def flatten(dataset_dir: Path) -> bool:
    """dataset/<sarmalayıcı>/<kaynaklar> yapısını dataset/<kaynaklar> yapar."""
    for w in dataset_dir.iterdir():
        if w.is_dir() and sources_in(w):
            logger.info(f'⚠️  Fazladan seviye: dataset/{w.name}/ → düzeltiliyor...')
            for item in list(w.iterdir()):
                target = dataset_dir / item.name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.move(str(item), str(target))
            shutil.rmtree(w, ignore_errors=True)
            return True
    return False


def extract(archive: Path, dest: Path) -> None:
    """Arşivi dest'e açar; mümkünse sistem aracını kullanır (daha hızlı)."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == '.tar':
        subprocess.run(['tar', '-xf', str(archive), '-C', str(dest)], check=True)
        return
    if shutil.which('unzip'):
        # -q sessiz, -o üzerine yaz. C ile yazılmış olduğu için 22k küçük
        # dosyada Python zipfile'dan belirgin hızlı.
        subprocess.run(['unzip', '-q', '-o', str(archive), '-d', str(dest)], check=True)
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)


def list_names(archive: Path) -> List[str]:
    if archive.suffix == '.tar':
        out = subprocess.run(['tar', '-tf', str(archive)], capture_output=True, text=True, check=True)
        return out.stdout.splitlines()
    with zipfile.ZipFile(archive) as zf:
        return zf.namelist()


def prepare(drive_root: Path, repo: Path) -> int:
    mydrive = Path('/content/drive/MyDrive')
    dataset_dir = repo / 'dataset'

    # 1) Zaten hazır mı? / yanlış seviyede mi?
    if dataset_dir.exists() and sources_in(dataset_dir):
        logger.info('ℹ️  dataset/ zaten doğru yapıda, hazırlık atlandı.')
    elif dataset_dir.exists() and any(dataset_dir.iterdir()) and flatten(dataset_dir):
        logger.info('✅ Yapı düzeltildi (yeniden açmaya gerek kalmadı).')
    else:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

        archive = find_archive(drive_root, mydrive)
        if not archive:
            logger.error(f'❌ Arşiv bulunamadı. Beklenen: {drive_root / "dataset" / ARCHIVE_NAMES[0]}')
            for d in (drive_root, drive_root / 'dataset'):
                if d.exists():
                    logger.error(f'📂 {d} → {sorted(q.name for q in d.iterdir())}')
            logger.error('Çözüm: dataset_colab.zip dosyasını yukarıdaki klasöre yükleyip '
                         'bu hücreyi tekrar çalıştırın.')
            return 1

        size_mb = archive.stat().st_size / 1e6
        logger.info(f'📦 Arşiv: {archive}  ({size_mb:.0f} MB)')
        if size_mb < 300:
            logger.warning(f'⚠️  Beklenen ~400 MB ama {size_mb:.0f} MB — yükleme bitmemiş olabilir.')

        # 2) Drive'dan yerel diske TEK SIRALI kopya (asıl hızlanma burada)
        t0 = time.time()
        # Depo ile aynı diskte tut (Colab'de /content) — sabit yol yazmıyoruz
        local_archive = repo.parent / archive.name
        if local_archive.exists():
            local_archive.unlink()
        shutil.copyfile(archive, local_archive)
        t_copy = time.time() - t0
        logger.info(f'   ⬇️  Yerel diske kopyalandı: {t_copy:.0f} sn '
                    f'({size_mb / max(t_copy, 0.1):.0f} MB/sn)')

        # 3) Yerelde aç (unzip varsa onunla)
        t0 = time.time()
        root = archive_root(list_names(local_archive))
        if root is None:
            logger.error('❌ Arşivde dataset klasörleri (augmented_train / *.yolo26) bulunamadı.')
            return 1
        logger.info(f"   arşiv kökü: '{root or '(sarmalayıcı yok)'}'")
        tmp = Path(tempfile.mkdtemp(dir=str(repo)))
        extract(local_archive, tmp)
        t_ex = time.time() - t0
        logger.info(f'   📂 Açıldı: {t_ex:.0f} sn')

        # 4) Kök neresi olursa olsun sonuç HER ZAMAN dataset/<kaynaklar>
        shutil.move(str(tmp / root if root else tmp), str(dataset_dir))
        shutil.rmtree(tmp, ignore_errors=True)
        local_archive.unlink(missing_ok=True)   # ~400 MB yer aç

    # 5) Doğrulama
    found = sorted(q.name for q in sources_in(dataset_dir))
    logger.info(f'\n📂 dataset/ içindeki kaynak klasörler ({len(found)}):')
    for f in found:
        logger.info(f'   - {f}')
    if not found:
        logger.error(f'❌ {dataset_dir} içinde kaynak klasör yok! '
                     f'Bulunanlar: {[d.name for d in dataset_dir.iterdir()][:10]}')
        return 1
    logger.info('\n✅ Dataset hazır.')
    return 0


def main():
    ap = argparse.ArgumentParser(description="Colab'de dataset'i Drive'dan hazırla")
    ap.add_argument('--drive-root', type=str,
                    default='/content/drive/MyDrive/SmartFarmStrawberryDisease',
                    help='Drive proje klasörü')
    ap.add_argument('--repo', type=str, default='/content/SmartFarmStrawberryDisease',
                    help='Depo kök dizini (dataset/ buraya açılır)')
    args = ap.parse_args()
    return prepare(Path(args.drive_root), Path(args.repo))


if __name__ == '__main__':
    raise SystemExit(main())
