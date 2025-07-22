# 🧠 CNN ile Beyin MR Görüntülerinin Sınıflandırılması

Bu proje, beyin tümörlerinin MRI (Manyetik Rezonans Görüntüleme) verileri üzerinden otomatik olarak sınıflandırılmasını hedeflemektedir. Derin öğrenme teknikleri kullanılarak sıfırdan geliştirilen bir Convolutional Neural Network (CNN) modeli ile Brain_Glioma, Brain_Menin (Meninjiyom) ve Brain_Tumor (genel tümör) sınıflarına ait beyin MR görüntüleri sınıflandırılmıştır.

<p align="center">
  <img src="test_dogrulugu.png" alt="Test Doğruluğu" width="300"/>
</p>

> 📌 **Test Doğruluğu: %99.01** — Model, test verisi üzerinde oldukça yüksek bir başarıya ulaşmıştır.

## 📁 Proje İçeriği

- CNN mimarisi ile MRI görüntülerinin sınıflandırılması
- Görüntü işleme ve veri artırma (data augmentation)
- Eğitim, doğrulama ve test bölmeleriyle eğitim stratejisi
- Performans metrikleri ile değerlendirme (accuracy, confusion matrix)
- Colab üzerinde çalışan kaynak kod bağlantısı

## 📊 Kullanılan Veri Seti

Veri seti: [Brain Cancer - MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)

- Toplam Görüntü Sayısı: 6056
- Sınıflar:
  - Brain_Glioma: 2004 görüntü
  - Brain_Menin (Meninjiyom): 2004 görüntü
  - Brain_Tumor: 2048 görüntü
- Görüntüler: Gri tonlamalı (grayscale), 512x512 piksel
- Yayım Tarihi: 5 Ağustos 2024
- Derleyen: Md Mizanur Rahman

### 🔀 Veri Bölme Oranları

- Eğitim verisi: %70 → 4240 görüntü
- Doğrulama verisi: %15 → 908 görüntü
- Test verisi: %15 → 908 görüntü

## 🛠️ Görüntü İşleme ve Dönüştürmeler

**Eğitim verisi için:**

- Grayscale'e çevirme
- Yeniden boyutlandırma (224x224)
- Yatay çevirme (%50 olasılık)
- Rasgele dönüş (±15°)
- Kaydırma, perspektif bozulması
- Gauss bulanıklığı
- Normalize (mean=0.5, std=0.5)

**Doğrulama/Test için:**

- Sadece yeniden boyutlandırma ve normalize işlemleri

## 🧠 Model Mimarisi

Sıfırdan oluşturulan CNN mimarisi şu bileşenlerden oluşmaktadır:

- 4 adet Conv2d + BatchNorm + ReLU + MaxPool katmanı
- Adaptive Average Pooling → Flatten
- 2 adet Fully Connected (Dense) katman
- Dropout (0.3) ile overfitting'e karşı önlem

### 📌 Örnek Katman Yapısı

```python
Conv2d(1, 32, 3, padding=1)
BatchNorm2d(32)
ReLU()
MaxPool2d(2)
...
Linear(256, 128)
Dropout(0.3)
Linear(128, 3)  # 3 sınıf çıkışı
🎯 Eğitim Stratejisi
Framework: PyTorch

Loss Function: CrossEntropyLoss

Optimizer: Adam (lr=0.001)

Epochs: 50

Batch Size: 32

Early Stopping: 10 epoch boyunca gelişme olmazsa durdurma

ReduceLROnPlateau: Val loss gelişmediğinde öğrenme oranını %50 azalt

📈 Performans
Modelin doğruluğu ve kaybı her epoch sonunda izlenmiş ve en iyi doğrulama başarımı elde edildiğinde model ağırlıkları kaydedilmiştir.
