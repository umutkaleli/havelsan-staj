# Deniz Üstü Nesne Takibi ve Risk Analizi Sistemi

## Özet
Bu proje, gemi kamerası perspektifinden deniz üstü nesneleri tespit ederek, bu nesnelerin hareketlerini ve çarpışma riskini analiz etmeye yönelik bir video işleme sistemidir.Nesne tespiti, optik akış ile hız tahmini, homografi ile gerçek dünya koordinat dönüşümü ve risk analizi algoritmaları bir arada kullanılmıştır.

## Kullanılan Algoritmalar ve Fonksiyonlar

- **YOLOv8 ile Nesne Tespiti:**
	- Video akışındaki deniz üstü nesneleri tespit etmek için Ultralytics YOLOv8 modeli kullanılır.
	- Sadece belirli sınıflar (ör. ship, boat, fishing_boat) risk analizine dahil edilir.
	- Diğer nesne tespit modelleride kullanılabilir.

- **Homografi ile Koordinat Dönüşümü:**
	- Kamera görüntüsündeki piksel koordinatları, gerçek dünya (metre cinsinden) koordinatlara dönüştürülmeye çalışılmıştır.
	- `px_to_m_homography` fonksiyonu bu dönüşümü sağlar.

- **Optik Akış ile Kendi Hız Tahmini:**
	- Kendi gemimizin hız vektörü, optik akış ROI bölgesinde Lucas-Kanade yöntemiyle tahmin edilir.
	- `estimate_ego_velocity` fonksiyonu kullanılır.

- **Nesne Takibi ve Hız Hesabı:**
	- Her nesne için alt-orta noktası homografi ile gerçek dünyaya çevrilir ve geçmiş pozisyonlardan hız vektörü EMA ile yumuşatılarak hesaplanır.
	- `update_track` fonksiyonu bu işlemleri yapar.

- **Risk Analizi (TCPA/DCPA):**
	- Kendi gemimiz ve hedef nesne arasındaki en yakın yaklaşım zamanı (TCPA) ve mesafesi (DCPA) hesaplanır.
	- Göreceli hız ve konum vektörleriyle çarpışma riski değerlendirilir.
	- `calculate_risk` fonksiyonu, TCPA, DCPA ve yaklaşma yönünü birleştirerek 0-100 arası bir risk skoru üretir.

- **Sonuçların Görselleştirilmesi:**
	- Her nesne için risk skoruna göre kutu rengi (kırmızı, sarı, yeşil) ve detaylı bilgi video üzerine yazılır.

## Fonksiyon Açıklamaları

- `px_to_m_homography(x_px, y_px, M)`: Pikselden metreye homografi dönüşümü.
- `estimate_ego_velocity(frame_gray, prev_gray, M, fps)`: Optik akış ile kendi hızını tahmin eder.
- `update_track(tracks, obj_id, cls_name, bbox, frame_idx, fps, M)`: Nesne takibi ve hız güncellemesi.
- `calculate_risk(p_own, v_own, p_target, v_target)`: TCPA/DCPA ve yaklaşma yönü ile risk skoru hesaplar.
- `color_for_risk(score)`: Risk skoruna göre kutu rengi döndürür.

## Zorluklar, Öğrenimler ve Sonuçlar

Bu projenin geliştirme sürecinde, teoride basit görünen birçok konseptin pratikte ne kadar karmaşık olabileceğini deneyimledim. Özellikle görüntü işleme ile fiziksel dünya arasında bir köprü kurmaya çalışırken karşılaşılan zorluklar, projenin en öğretici kısımlarını oluşturdu.

- **Gerçek Dünya Koordinatlarına Çevirmenin Zorluğu:**
	- Homografi dönüşümü, kağıt üzerinde mükemmel bir çözüm gibi dursa da, kamera kalibrasyonundaki en ufak bir hata veya geminin yalpalaması gibi faktörler, mesafe ve hız tahminlerinde ciddi sapmalara yol açabiliyor. Görüntüdeki bir pikselin, denizin ortasında kaç metreye denk geldiğini kesin olarak söylemek, neredeyse imkansız.

- **Hız Tahminindeki Belirsizlikler:**
	- **Optik Akış:** Deniz yüzeyinin sürekli değişen doğası (dalgalar, yansımalar), optik akış tabanlı hız tahminini oldukça zorluyor. Yöntem, zaman zaman suyun hareketini geminin hızı olarak algılayarak yanıltıcı sonuçlar üretebiliyor.
	- **Konum Farkından Hesaplama:** Nesnelerin takip edilen konumları arasındaki farktan hız hesaplamak, homografi ve takip sistemindeki hataları biriktirerek güvenilirliği düşürüyor.

- **Boundary Box Boyut Analiziyle Yaklaşma Tespiti:**
	- Bir nesnenin yaklaşıp yaklaşmadığını anlamak için sınırlayıcı kutusunun (bounding box) boyutundaki değişimi analiz etmeyi denedim. Mantıken, kutu büyüyorsa nesne yaklaşıyor demektir. Ancak bu yöntem, nesnenin farklı açılarla dönmesi veya tespit modelinin anlık olarak kutuyu farklı boyutlarda çizmesi gibi nedenlerle kararlı çalışmadı. Bir karede büyük çizilen bir kutu, bir sonraki karede küçük çizilebiliyor ve bu da sürekli yanlış sonuçlara çıkabiliyordu.

- **Nesne Takibindeki Tutarsızlıklar:**
	- YOLOv8'in nesne takipçisi genel olarak başarılı olsa da, özellikle nesneler birbirine yaklaştığında veya kısa süreliğine kaybolup yeniden ortaya çıktığında kimlik (ID) atamalarında hatalar yapabiliyor. Bu durum, bir nesnenin hız ve risk geçmişinin tamamen yanlış hesaplanmasına neden olabiliyor.

- **Matematiksel Analizin Sınırları:**
	- Bu proje, TCPA/DCPA gibi matematiksel formüllerin, üzerine kurulu olduğu verilerin (konum, hız) doğruluğuna ne kadar bağımlı olduğunu net bir şekilde gösterdi. Girdi verilerindeki en ufak bir gürültü veya hata, risk skorunu tamamen anlamsız hale getirebiliyor. Sadece 2D analiz yapılması ve akıntı, rüzgar gibi çevresel faktörlerin hesaba katılmaması da bu modelin gerçek dünya karmaşıklığı karşısındaki sınırlarını ortaya koyuyor.

### Sonuç
Bu proje, deniz üstü risk analizini yalnızca görüntü işleme ve matematiksel modellerle çözmenin pratikteki zorluklarını ortaya koyan önemli bir deneyim olmuştur. Homografi, optik akış ve TCPA/DCPA gibi belirtilen yöntemleri uygulamama rağmen, "Zorluklar" bölümünde detaylandırılan nedenlerden ötürü tam olarak doğru ve güvenilir sonuçlar elde etmek mümkün olmamıştır.

Bu çalışma göstermiştir ki, bu problemi sadece hazır modeller ve algoritmalarla, özel bir model eğitimi olmaksızın çözmek bence oldukça zordur. Sistem, ideal ve stabil koşullarda zaman zaman faydalı öngörüler sunabilse de, deniz ortamının dinamik ve değişken doğası karşısında sıkça hata yapmaktadır. Bu nedenle, kritik bir karar destek sistemi olarak kullanılması mevcut haliyle uygun değildir. Sonuç olarak, bu projenin en büyük kazanımı, problemin karmaşıklığını ve sadece sensör verisine dayalı veya bu verilerle eğitilmiş özel yapay zeka modelleriyle daha sağlam çözümler üretilebileceğini anlamak olmuştur.
