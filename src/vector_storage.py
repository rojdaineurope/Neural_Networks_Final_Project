import pandas as pd
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Veriyi Yükle (JSON Lines formatında olduğu için lines=True)
print("Film detayları okunuyor...")
movies_df = pd.read_json('IMDB_movie_details.json', lines=True)

# 2. Sadece özeti (plot_synopsis) olanları al
# Bilgisayarını yormamak için ilk 1000 filmle başlayalım
movies_sample = movies_df[movies_df['plot_synopsis'].notna()].head(1000)

docs = []
for _, row in movies_sample.iterrows():
    # Sütun ismini güvenli bir şekilde alalım
    m_id = str(row['movie_id'])
    # Eğer movie_name yoksa movie_id'yi isim olarak kullanalım
    m_title = row.get('movie_name', m_id) 
    
    docs.append(Document(
        page_content=row['plot_synopsis'],
        metadata={"movie_id": m_id, "title": m_title}
    ))

# 3. Metni Parçalara Böl (Ajanın okuyabileceği küçük parçalar)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 4. Vektörleştirme Modeli (HuggingFace)
print("Embedding modeli yükleniyor...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Vektör Veritabanını (ChromaDB) Oluştur
print(f"{len(split_docs)} metin parçası vektörleştiriliyor... (Birkaç dakika sürebilir)")
vector_db = Chroma.from_documents(
    documents=split_docs, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("✅ Başarıyla tamamlandı! 'chroma_db' klasörü projenin hafızası olarak hazır.")