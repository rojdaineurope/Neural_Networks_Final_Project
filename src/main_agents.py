import os
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. .env dosyasındaki değişkenleri yükle
load_dotenv() 
# 1. Ayarlar (Groq API Key'ini buraya yaz veya .env'den al)
# 2. Anahtarı sistemden çek
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 2. Hafızayı Bağla (ChromaDB klasörünü okur)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def identify_spoiler(comment, movie_id):
    # --- AJAN 1: BİLGİ GETİRİCİ (Retriever) ---
    # Sadece ilgili filme ait özeti çekmek için filter kullanıyoruz
    results = vector_db.similarity_search(comment, k=2, filter={"movie_id": movie_id})
    
    if not results:
        return "Film özeti hafızada bulunamadı."
    
    movie_context = "\n".join([res.page_content for res in results])

    # --- AJAN 2: ANALİZ UZMANI (Llama-3) ---
    prompt = f"""
    Sen bir Film Spoiler Tespit Uzmanısın. Aşağıdaki film özeti bilgisini kullanarak yorumu analiz et.

    FİLM ÖZETİ (GERÇEK BİLGİ):
    {movie_context}

    KULLANICI YORUMU:
    {comment}

    GÖREV:
    Bu yorum, film özetiyle kıyaslandığında kritik bir olay (ölüm, son, büyük sürpriz) ifşa ediyor mu?
    1. Cevabına 'KARAR: SPOILER' veya 'KARAR: NORMAL' diyerek başla.
    2. Nedenini 1 cümleyle açıkla (Örn: 'Çünkü ana karakterin kaderinden bahsediyor').
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", # En yeni ve en zeki model budur! modeli değiştirdik
    )
    
    return chat_completion.choices[0].message.content

# --- TEST ET ---
# Örnek: Film ID'sini ve yorumu ver (Senin datasetinden bir ID seçebilirsin)
# print(identify_spoiler("John dies at the end of the mission!", "tt0105112"))

# --- TEST BÖLÜMÜ ---
if __name__ == "__main__":
    # Test için bir yorum ve o filme ait doğru bir ID
    # Eğer filmin sonunda birinin öldüğünü biliyorsan onu yaz
    # Madem hafızada bu film var, gerçek bir bilgiyle test edelim
    sonuc = identify_spoiler("Miller's brother is killed during the initial assassination attempt!", "tt0105112")
    print("\n--- AJAN ANALİZİ ---")
    print(sonuc)

    # Hangi filmler hafızada var, bir bakalım?
# Bunu main_agents.py'nin en altına, test kısmına ekle
#print("\nHafızadaki bazı filmler:")
#print(vector_db.get(limit=5)['metadatas'])