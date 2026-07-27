# models/

Eğitilmiş model dosyasını (`best.pt`) buraya koyun.

Colab'de eğitim bitince model şuraya kaydedilir:
```
MyDrive/SmartFarmStrawberryDisease/best_models/best_<kosu_adi>.pt
```

Drive'dan indirip bu klasöre `best.pt` adıyla kopyalayın:
```
models/best.pt
```

Farklı bir yol kullanmak isterseniz `MODEL_PATH` ortam değişkenini ayarlayın:
```bash
set MODEL_PATH=C:\yol\best.pt   # Windows
```

> Model dosyaları git'e eklenmez (.gitignore'da `*.pt` var).
