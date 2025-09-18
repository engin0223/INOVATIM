# ❤️ NabızAI – Derin Öğrenme ile Hiyerarşik Aritmi Sınıflandırması

Bu veri havuzu, **MIT-BIH Aritmi Veri Seti** kullanılarak aritmi tespiti ve alt tip sınıflandırması için **hiyerarşik bir derin öğrenme işlem hattı** sağlar.

Sistem önce **Normal ve Aritmi** (ikili sınıflandırma) arasında ayrım yapar, ardından aritmik atımları ilgili **alt tiplerine** göre sınıflandırır.

---

## 📌 Özellikler
- **Adım 1:** İkili sınıflandırma (Normal ve Aritmi)
- **Adım 2:** Çok sınıflı aritmi alt tipi sınıflandırması
- **Hiyerarşik tahmin hattı:** Uçtan uca EKG atım sınıflandırması için her iki modeli birleştirir
- **Değerlendirme Ölçütleri:** Doğruluk, Kesinlik, Hatırlama, F1 puanı (sınıf başına ve genel)
- **Görselleştirme:** Karışıklık matrisi ve performans grafikleri

---

## 📂 Veri Seti
Proje, önceden işlenmiş **MIT-BIH Aritmi Veri Setini** kullanmaktadır:
- `X` → `[N x SeqLen x Özellikler]` şeklindeki EKG atımları
- `Y` → Atım etiketleri (ANSI/AAMI EC57 standardı)

> ⚠️ Boyut kısıtlamaları nedeniyle, ham veri seti **Bu depoda yer almamaktadır**.
[PhysioNet MIT-BIH Aritmi Veritabanı](https://physionet.org/content/mitdb/1.0.0/) adresinden indirebilirsiniz.

---

## ⚙️ Gereksinimler
- MATLAB R2021b veya üzeri
- Derin Öğrenme Araç Kutusu
- İstatistik ve Makine Öğrenmesi Araç Kutusu

---

## 🚀 Kullanım
1. Bu deponun klonunu oluşturun:
```bash
git clone https://github.com/<kullanıcı-adınız>/NabizAI.git
cd NabizAI
