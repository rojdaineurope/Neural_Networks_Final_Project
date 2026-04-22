import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Gerekli dil paketini indirelim
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. HTML etiketlerini temizle (ör: <br>)
    text = re.sub(r'<.*?>', '', text)
    # 2. Sadece harfleri tut (rakam ve noktalama işaretlerini at)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Küçük harfe çevir
    text = text.lower().strip()
    # 4. Gereksiz kelimeleri (stopwords) çıkar
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

# VERİYİ YÜKLEME
print("Büyük veri okunuyor...")
# Belleği yormamak için ilk 20.000 satırı alalım
df = pd.read_json('IMDB_reviews.json', lines=True, chunksize=20000)
sample_df = next(df)

print(f"Toplam {len(sample_df)} satır temizleniyor...")
tqdm.pandas() # İlerleme çubuğu için
sample_df['cleaned_review'] = sample_df['review_text'].progress_apply(clean_text)

# İhtiyacımız olan sütunları seçelim
# 'is_spoiler' hedef değişkenimiz (label)
processed_df = sample_df[['movie_id', 'is_spoiler', 'cleaned_review']]

# TEMİZLENMİŞ VERİYİ KAYDET
processed_df.to_csv('cleaned_reviews.csv', index=False)
print("İşlem tamam! 'cleaned_reviews.csv' dosyası oluşturuldu.")

import os

output_file = 'cleaned_reviews.csv'

# Eğer dosya daha önce oluşturulmadıysa temizleme işlemini yap
if not os.path.exists(output_file):
    print("Temizlenmiş dosya bulunamadı. İşlem başlatılıyor...")
    # ... (burada senin temizleme kodların, df.to_csv vb. olacak) ...
    processed_df.to_csv(output_file, index=False)
else:
    print(f"'{output_file}' zaten mevcut, direkt yükleniyor...")
    processed_df = pd.read_csv(output_file)

# Sonuçları her durumda görmek için buraya yazıyoruz
print("\n--- Veri Seti Spoiler Dağılımı ---")
print(processed_df['is_spoiler'].value_counts())