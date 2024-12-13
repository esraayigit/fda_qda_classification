### FDA ve QDA Uygulamaları ile Veri Analizi
Bu proje, Fisher Discriminant Analysis (FDA) ve Quadratic Discriminant Analysis (QDA) algoritmalarını kullanarak veri sınıflandırması yapmaktadır. Wine veri seti üzerinde gerçekleştirilmiştir. Proje, PCA (Principal Component Analysis) ile veri boyutunun indirgenmesi, ardından FDA ve QDA yöntemleri ile sınıflandırma ve model performansının karşılaştırılmasını içermektedir.

## Adımlar:
### Veri Ön İşleme: 
Wine veri setinden ilgi çekici özellikler seçilip, standartlaştırıldı.

### PCA ile Boyut İndirgeme:
Özelliklerin boyutları 2D'ye indirgenerek verilerin görselleştirilmesi sağlandı.

### FDA ve QDA Modelleri:
Fisher Discriminant Analysis (FDA) kullanılarak iki sınıf arasındaki farkın maksimal olarak ayrılması sağlandı.
Quadratic Discriminant Analysis (QDA) kullanılarak her sınıf için kovaryans matrisi ile sınıflandırma yapıldı.

### Sonuçların Görselleştirilmesi: 
FDA ve QDA'nın discriminant skorları histogramlar şeklinde görselleştirildi.

### Model Performansı:
Her iki modelin performansı, Area Under ROC Curve (AUROC) metriği ile karşılaştırıldı.

## Kullanılan Yöntemler:
### PCA (Principal Component Analysis): 
Verilerin boyutunu indirgerken en fazla varyansı korur.

### FDA (Fisher Discriminant Analysis):
İki sınıf arasındaki farkı maksimize etmeyi amaçlar.

### QDA (Quadratic Discriminant Analysis): 
Sınıflar için kovaryans matrislerinin farklı olduğu varsayımına dayanır ve discriminant fonksiyonları kullanarak sınıflandırma yapar.
## Sonuçlar:
Model performansı AUROC değeri ile ölçülmüş ve her iki modelin başarımı karşılaştırılmıştır.
Bu proje, sınıflandırma problemlerine yönelik temel diskriminant analiz tekniklerinin nasıl uygulandığını ve performanslarını kıyaslamayı amaçlamaktadır.
