#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install geopandas


# In[ ]:


pip install mplcursors


# In[5]:


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[114]:


# Excel dosyasını oku
# CSV dosyasını okuma
df = pd.read_csv((r"D:\data\mgm\18.07.2023.csv"))
df_2 = pd.read_csv("D:\data\mgm\ekstremler.csv", encoding='latin1')
df_3 = pd.read_csv("D:\data\mgm\maksimum.csv", encoding='latin1')


# In[115]:


# 0. satırı silmek için:
df_3 = df_3.drop(0)
print(df_3)


# In[129]:


import pandas as pd
import difflib

# Örnek veri setleri
data1 = {
    'Merkezler': ['Ahmetli', 'Akhisar', 'Alasehir'],
    'Temmuz': [43.2, 45.2, 42.1],
    'Yillik': [43.2, 45.2, 42.7],
    'Tarih': ['7/1/2017', '7/2/2017', '6/20/2016']
}

data2 = {
    'IST. ADI': ['Ahmetli', '17111|CANAKKALE|BOZCAADA', '17112|CANAKKALE|CANAKKALE'],
    'TARIH (GMT)': ['7/20/2023', '7/20/2023', '7/20/2023'],
    'MAKSIMUM': [36.4, 32.1, 37.1]
}

df_3 = pd.DataFrame(data2)
df_1 = pd.DataFrame(data1)

# Sütun adlarını güncelleme
df_3.columns = ['Merkezler', 'TARIH (GMT)', 'MAKSIMUM']

# Sütun isimleri arasında benzerlikleri bulup eşleşen satırları içeren yeni bir veri seti oluşturma işlemi
matched_rows = []

for col1 in df_1['Merkezler']:
    for col3 in df_3['Merkezler']:
        # İki sütun ismi arasındaki benzerlik oranını bulma
        similarity_ratio = difflib.SequenceMatcher(None, col1, col3).ratio()
        
        # Benzerlik oranı eşik değerinden büyükse (örn. 0.5), bu sütun isimleri eşleşiyor kabul edilir
        if similarity_ratio > 0.5:
            matched_rows.extend(df_3[df_3['Merkezler'] == col3].to_dict(orient='records'))

# Eşleşen satırları içeren yeni veri setini oluşturma
matched_df = pd.DataFrame(matched_rows)

print(matched_df)


# In[8]:


def turkish_to_english(text):
    turkish_letters = "çÇğĞıİöÖşŞüÜ"
    english_letters = "cCgGiIoOsSuU"
    translation_table = str.maketrans(turkish_letters, english_letters)
    return text.translate(translation_table)

# CSV dosyasını oku ve Türkçe harf içeren ilçe isimlerini İngilizce karakterlere dönüştür
df = pd.read_csv((r"D:\data\mgm\18.07.2023.csv"))
df['Merkezler'] = df['Merkezler'].apply(turkish_to_english)

# Verileri printle
print(df)


# In[9]:


veri_cerceveleri = {}  # Veri çerçevelerini saklamak için boş bir sözlük oluştur

sütun_adi = 'Merkezler'  # İlgili sütun adını belirtin

for kelime in df[sütun_adi].unique():
    kelime = kelime.lower()  # Kelimeyi küçük harflere çevir
    veri_cerceveleri[kelime] = df[df[sütun_adi].str.lower() == kelime]


# In[10]:


veri_cerceveleri_2 = {}  # İkinci veri çerçevelerini saklamak için boş bir sözlük oluştur

sütun_adı_2 = 'Merkezler'  # İkinci veri setindeki ilgili sütun adını belirtin

for kelime_2 in df_2[sütun_adı_2].unique():
    kelime_2 = kelime_2.lower()  # Kelimeyi küçük harflere çevir
    kelime_2 = kelime_2.replace(" ", "_")  # Boşlukları alt çizgi ile değiştir
    veri_cerceveleri_2[kelime_2] = df_2[df_2[sütun_adı_2].str.lower() == kelime_2]

veri_cerceveleri_2['bandirma']

# Örnek kullanım: bornova = veri_cerceveleri_2['bornova']


# In[11]:


tum_degerler = {}  # Tüm satırlara ait değerleri saklamak için boş bir sözlük oluştur

for kelime, veri_cercevesi in veri_cerceveleri.items():
    degerler = {}
    for sütun in veri_cercevesi.columns[1:]:  # İlk sütunu hariç al
        degerler[sütun] = veri_cercevesi[sütun].values[0]
    tum_degerler[kelime] = degerler

# Örnek kullanım: tum_degerler['cigli'] ile Cigli'ye ait bütün sütun değerlerine erişebilirsiniz


# In[12]:


tum_degerler_2 = {}  # Tüm satırlara ait değerleri saklamak için boş bir sözlük oluştur

for kelime_2, veri_cercevesi_2 in veri_cerceveleri_2.items(): 
    degerler_2 = {} 
    
    if not veri_cercevesi_2.empty:  # Veri çerçevesi boş değilse işlemleri yap
        for sütun_2 in veri_cercevesi_2.columns[1:]:  # İlk sütunu hariç al
            degerler_2[sütun_2] = veri_cercevesi_2[sütun_2].values[0]
        
    tum_degerler_2[kelime_2] = degerler_2

# Güncellenen verileri kontrol etme
# print(tum_degerler_2)


# In[13]:


# tum_degerler_2 sözlüğünü DataFrame'e dönüştürme
tum_degerler_2_df = pd.DataFrame.from_dict(tum_degerler_2, orient='index', columns=['Merkez', 'Ekstrem'])

if 'söke' in tum_degerler_2:
    tum_degerler_2['soke'] = tum_degerler_2.pop('söke')

if 'çine' in tum_degerler_2:
    tum_degerler_2['cine'] = tum_degerler_2.pop('çine')
    
if 'kösk' in tum_degerler_2:
    tum_degerler_2['kosk'] = tum_degerler_2.pop('kösk')
    
if 'gördes' in tum_degerler_2:
    tum_degerler_2['gordes'] = tum_degerler_2.pop('gördes')
    
if 'kirkagaç' in tum_degerler_2:
    tum_degerler_2['kirkagac'] = tum_degerler_2.pop('kirkagaç')
    
if 'sarigöl' in tum_degerler_2:
    tum_degerler_2['sarigol'] = tum_degerler_2.pop('sarigöl')
    
if 'gölmarmara' in tum_degerler_2:
    tum_degerler_2['golmarmara'] = tum_degerler_2.pop('gölmarmara')
    
tum_degerler_2['yunusemre']
    
# tum_degerler_2


# In[14]:


tum_degerler['yunusemre']
#tum_degerler['edremit']


# In[15]:


import pandas as pd

extreme_merkezler = []
eks_yakin_merkezler = []
data = []

for kelime, degerler in tum_degerler.items():
    if kelime in tum_degerler_2:
        yillik_sicaklik = tum_degerler_2[kelime]['Yillik']
        max_sicaklikler = [degerler[f'Max{i}'] for i in range(1, 6)]

        max_sicaklik = max(max_sicaklikler)
        print('Güncel Sıcaklık: ', max_sicaklik)
        print('Eski Ekstrem Sıcaklık: ', yillik_sicaklik)

        if max_sicaklik >= yillik_sicaklik:
            print(f"{kelime} bölgesinde ekstrem sıcaklık değerleri tespit edildi.")
            extreme_merkezler.append(kelime)
            data.append({'Merkez': kelime, 'Ekstrem': True, 'Eks_Yakin': False, 'Max_Sicaklik': max_sicaklik})
        elif abs(max_sicaklik - yillik_sicaklik) < 2:
            print(f"{kelime} bölgesinde ekstrem sıcaklık değerlerine yakın bir durum tespit edildi.")
            eks_yakin_merkezler.append(kelime)
            data.append({'Merkez': kelime, 'Ekstrem': False, 'Eks_Yakin': True, 'Max_Sicaklik': max_sicaklik})
        else:
            print(f"{kelime} bölgesinde ekstrem sıcaklık değerleri tespit edilmedi.")
            data.append({'Merkez': kelime, 'Ekstrem': False, 'Eks_Yakin': False, 'Max_Sicaklik': max_sicaklik})
    else:
        # print(f"{kelime} bölgesine ait yıllık sıcaklık verisi bulunamadı.")
        data.append({'Merkez': kelime, 'Ekstrem': False, 'Eks_Yakin': False, 'Max_Sicaklik': max_sicaklik})

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Filter extreme and near-extreme regions separately
df_extreme_deger = df[df['Ekstrem'] == True]
df_eks_yakin_deger = df[df['Eks_Yakin'] == True]

# Display the tables
print("Extreme Regions:")
print(df_extreme_deger)

print("\nNear-Extreme Regions:")
print(df_eks_yakin_deger)


# In[16]:


df_extreme = pd.DataFrame(df_extreme_deger)
df_eks_yakin = pd.DataFrame(df_eks_yakin_deger)

# DataFrame'i printle
print("DataFrame - Extreme:")
print(df_extreme)

print("\nDataFrame - Near-Extreme:")
print(df_eks_yakin)



# In[17]:


print('Ekstrem Merkezleri:', '\n', extreme_merkezler,'\n')
print('Ekstreme Yakın Merkezleri:', '\n', eks_yakin_merkezler)


# In[18]:


data_filtered = [d for d in data if d['Ekstrem']]

data_dict_filtered = {
    'Merkez': [d['Merkez'] for d in data_filtered],
    'Ekstrem': [d['Ekstrem'] for d in data_filtered]
}

df_filtered = pd.DataFrame(data_dict_filtered)

# 'Merkez' sütunundaki "bandirma" kelimesini "Bandırma" ve "sindirgi" kelimesini "Sındırgı" olarak değiştir
df_filtered['Merkez'] = df_filtered['Merkez'].replace({'bandirma': 'Bandırma', 'marmaraadasi': 'Marmara','sindirgi':'Sındırgı', 'izmir': 'Konak',
                                                      'ayvalik': 'Ayvalık', 'ayvacik-canakkale': 'Ayvacık', 'yenice-canakkale': 'Yenice', 
                                                      'torbali': 'Torbalı', 'bayindir': 'Bayındır', 'narlidere': 'Narlıdere', 'karsiyaka': 'Karşıyaka',
                                                      'kinik': 'Kınık', 'cine': 'Çine', 'kusadasi': 'Kuşadası', 'soke': 'Söke', 'kirkagac': 'Kırkağaç',
                                                      'sarigol': 'Sarıgöl', 'koprubasi-manisa': 'Köprübaşı', 'saruhanli': 'Saruhanlı',
                                                    'bozcada': 'Bozcaada','altieylul': 'Balıkesir', 'efeler': 'Aydın','yunusemre': 'Manisa'}, regex=True)


print(df_filtered)


# In[19]:


import re
import seaborn as sns

def plot_with_extreme(center_name, center_data, extreme_value):
    plt.figure(figsize=(10, 6))
    
    tarihler = center_data.columns[1:]  # Tarih sütunlarını al
    
    min_sıcaklık_verileri = []
    max_sıcaklık_verileri = []
    tarihler_xticks = []
    
    for i, sütun in enumerate(tarihler):
        if sütun.startswith('Min'):
            min_sıcaklık_verileri.append(center_data.iloc[0, center_data.columns.get_loc(sütun)])
            max_sıcaklık_verileri.append(center_data.iloc[0, center_data.columns.get_loc(sütun.replace('Min', 'Max'))])
            if i > 0:
                tarih = re.findall(r'\d{2}\.\d{2}\.\d{2}', tarihler[i-1])  # Tarih değerini solundaki sütundan ayıkla
                if tarih:
                    tarihler_xticks.append(tarih[0])  # Tarih değerini listeye ekle
    
    if not min_sıcaklık_verileri:
        return
    
    # Seaborn stili ve ölçeği ayarla
    sns.set(style='whitegrid', palette='viridis')
    sns.set_context("notebook", font_scale=1.2)
    
    plt.title(f'Ekstrem Değer ile {center_name.capitalize()} Sıcaklık Grafiği')
    plt.xlabel('Tarih')
    plt.ylabel('Sıcaklık')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Ekstrem değeri yazdır
    plt.text(tarihler_xticks[-1], extreme_value, f'Ekstrem: {extreme_value}', ha='right', va='bottom',
             color='black', backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.7))
    
    # Sıcaklık değerleri için lineplot çiz
    plt.plot(tarihler_xticks, min_sıcaklık_verileri, marker='o', linestyle='-', color='b', label='Min Sıcaklık')
    plt.plot(tarihler_xticks, max_sıcaklık_verileri, marker='o', linestyle='-', color='g', label='Max Sıcaklık')

    # Ekstrem değeri göstermek için bir çizgi çizin
    plt.axhline(y=extreme_value, color='r', linestyle='--', linewidth=2, label='Ekstrem Değer')
    y_min = min(min_sıcaklık_verileri + [extreme_value]) - 1
    y_max = max(max_sıcaklık_verileri + [extreme_value]) + 1
    plt.ylim(y_min, y_max)  # Y eksenini minimum ve maksimum sıcaklık değerleri arasında ayarlayın

    # Ekstrem değeri aşan noktaların arka planını kırmızı yap
    for tarih, min_sicaklik, max_sicaklik in zip(tarihler_xticks, min_sıcaklık_verileri, max_sıcaklık_verileri):
        if min_sicaklik > extreme_value or max_sicaklik > extreme_value:
            plt.axvspan(tarih, tarih, facecolor='red', alpha=0.2)
    
    # Etiketleri noktanın üzerine yerleştirin
    for x, y_max, y_min in zip(tarihler_xticks, max_sıcaklık_verileri, min_sıcaklık_verileri):
        plt.text(x, y_max, str(y_max), ha='center', va='bottom', color='white', backgroundcolor='black', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))
        plt.text(x, y_min, str(y_min), ha='center', va='top', color='white', backgroundcolor='black', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.7))
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Örnek kullanım: aliaga için grafiği oluştur
center_name = 'cigli'
center_data = veri_cerceveleri[center_name]
extreme_value = tum_degerler_2[center_name]['Yillik']

plot_with_extreme(center_name, center_data, extreme_value)


# In[20]:


# CSV dosyasını okuma
df = pd.read_csv(r"D:\data\mgm\istasyonlar.csv")

# Veriyi görüntüleme
# print(df)


# In[21]:


# Çanakkale bölgesinin lat ve lon sınırları
min_lat = 39.600
max_lat = 40.400
min_lon = 26.000
max_lon = 27.400

# Çanakkale bölgesi içindeki istasyonları seçme
canakkale_istasyonlar = df[(df['Lat'] >= min_lat) & (df['Lat'] <= max_lat) & (df['Lon'] >= min_lon) & (df['Lon'] <= max_lon)]

# Seçilen istasyonları görüntüleme
# print(canakkale_istasyonlar)


# In[91]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')

# CSV dosyasını okuma
df = pd.read_csv(r"D:\data\mgm\istasyonlar.csv")

# İzmir bölgesinin lat ve lon sınırları
izmir_min_lat = 37.868458
izmir_max_lat = 38.899965
izmir_min_lon = 25.757508
izmir_max_lon = 27.210144

# İzmir bölgesi içindeki istasyonları seçme
izmir_istasyonlar = df[(df['Lat'] >= izmir_min_lat) & (df['Lat'] <= izmir_max_lat) & (df['Lon'] >= izmir_min_lon) & (df['Lon'] <= izmir_max_lon)]

# Seçilen İzmir istasyonlarını görüntüleme
# print("İzmir İstasyonlar:")
# print(izmir_istasyonlar)
# print()

# Manisa bölgesinin lat ve lon sınırları
manisa_min_lat = 37.982692
manisa_max_lat = 39.225856
manisa_min_lon = 27.436439
manisa_max_lon = 28.782433

# Manisa bölgesi içindeki istasyonları seçme
manisa_istasyonlar = df[(df['Lat'] >= manisa_min_lat) & (df['Lat'] <= manisa_max_lat) & (df['Lon'] >= manisa_min_lon) & (df['Lon'] <= manisa_max_lon)]

# Seçilen Manisa istasyonlarını görüntüleme
# print("Manisa İstasyonlar:")
# print(manisa_istasyonlar)
# print()

# Çanakkale bölgesinin lat ve lon sınırları
canakkale_min_lat = 39.480753
canakkale_max_lat = 40.312442
canakkale_min_lon = 25.686228
canakkale_max_lon = 27.242763

# Çanakkale bölgesi içindeki istasyonları seçme
canakkale_istasyonlar = df[(df['Lat'] >= canakkale_min_lat) & (df['Lat'] <= canakkale_max_lat) & (df['Lon'] >= canakkale_min_lon) & (df['Lon'] <= canakkale_max_lon)]

# Seçilen Çanakkale istasyonlarını görüntüleme
# print("Çanakkale İstasyonlar:")
# print(canakkale_istasyonlar)
# print()

# Aydın bölgesinin lat ve lon sınırları
aydin_min_lat = 37.358373
aydin_max_lat = 38.292651
aydin_min_lon = 26.513294
aydin_max_lon = 28.855722

# Aydın bölgesi içindeki istasyonları seçme
aydin_istasyonlar = df[(df['Lat'] >= aydin_min_lat) & (df['Lat'] <= aydin_max_lat) & (df['Lon'] >= aydin_min_lon) & (df['Lon'] <= aydin_max_lon)]

# Seçilen Aydın istasyonlarını görüntüleme
# print("Aydın İstasyonlar:")
# print(aydin_istasyonlar)
# print()

# Balıkesir bölgesinin lat ve lon sınırları
balikesir_min_lat = 38.669117
balikesir_max_lat = 40.958249
balikesir_min_lon = 26.523692
balikesir_max_lon = 29.148618

# Balıkesir bölgesi içindeki istasyonları seçme
balikesir_istasyonlar = df[(df['Lat'] >= balikesir_min_lat) & (df['Lat'] <= balikesir_max_lat) & (df['Lon'] >= balikesir_min_lon) & (df['Lon'] <= balikesir_max_lon)]

# Seçilen Balıkesir istasyonlarını görüntüleme
# print("Balıkesir İstasyonlar:")
# print(balikesir_istasyonlar)
# print()

# JSON dosyasını okuma
gdf = gpd.read_file("D:\data\mgm\gadm41_TUR_2.json")

# İzmir il sınırları
izmir_il_sınırlar = gdf[gdf['NAME_1'] == 'Izmir']

# Manisa il sınırları
manisa_il_sınırlar = gdf[gdf['NAME_1'] == 'Manisa']

# Çanakkale il sınırları
canakkale_il_sınırlar = gdf[gdf['NAME_1'] == 'Çanakkale']

# Aydın il sınırları
aydin_il_sınırlar = gdf[gdf['NAME_1'] == 'Aydin']

# Balıkesir il sınırları
balikesir_il_sınırlar = gdf[gdf['NAME_1'] == 'Balikesir']

# İzmir ilçeleri
izmir_il_ilceler = gdf[gdf['NAME_1'] == 'Izmir']

# Manisa ilçeleri
manisa_il_ilceler = gdf[gdf['NAME_1'] == 'Manisa']

# Çanakkale ilçeleri
canakkale_il_ilceler = gdf[gdf['NAME_1'] == 'Çanakkale']

# Aydın ilçeleri
aydin_il_ilceler = gdf[gdf['NAME_1'] == 'Aydin']

# Balıkesir ilçeleri
balikesir_il_ilceler = gdf[gdf['NAME_1'] == 'Balikesir']

# Tüm ilçeleri birleştirme
tum_ilceler = pd.concat([izmir_il_ilceler, manisa_il_ilceler, canakkale_il_ilceler, aydin_il_ilceler, balikesir_il_ilceler])

# Haritayı çizdirme
fig, ax = plt.subplots(figsize=(24, 24))
izmir_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='blue')
manisa_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='green')
canakkale_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='red')
aydin_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='purple')
balikesir_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='orange')

ax.set_aspect('equal')

# Tüm ilçeleri çizdirme ve rengini değiştirme
tum_ilceler.boundary.plot(ax=ax, linewidth=0.5, color='black')
tum_ilceler.plot(ax=ax, color='lightblue')

# İlçe isimlerini yazdırma
for idx, row in tum_ilceler.iterrows():
    ax.annotate(text=row['NAME_2'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight= 'bold')


import unicodedata

izmir_ilceler_listesi = ['Aliağa', 'Balçova', 'Bayındır', 'Bayraklı', 'Bergama', 'Beydağ', 'Bornova',
                        'Buca', 'Çeşme', 'Çiğli', 'Dikili', 'Foça', 'Gaziemir', 'Güzelbahçe', 'Karabağlar',
                        'Karaburun', 'Karşıyaka', 'Kemalpaşa', 'Kınık', 'Kiraz', 'Konak', 'Menderes',
                        'Menemen', 'Narlıdere', 'Ödemiş', 'Seferihisar', 'Selçuk', 'Tire', 'Torbalı', 'Urla', 'Bayındır']

canakkale_ilceler_listesi = ['Çanakkale', 'Gökçeada', 'Ezine', 'Biga', 'Lapseki', 'Bozcaada', 'Gelibolu', 'Ayvacık', 
                             'Bayramiç', 'Çan', 'Eceabat', 'Yenice']

aydin_ilceler_listesi = ['Aydın', 'Bozdoğan', 'Buharkent', 'Kuşadası',
                         'Didim', 'Nazilli', 'Sultanhisar', 'Söke', 'Çine', 'Germencik', 'İncirliova', 
                         'Karacasu', 'Karpuzlu', 'Köşk', 'Kuyucak', 'Yenipazar']

manisa_ilceler_listesi = ['Manisa', 'Merkez', 'Yunusemre', 'Akhisar', 'Salihli', 'Turgutlu', 'Alaşehir', 'Köprübaşı', 
                          'Soma', 'Ahmetli', 'Gölmarmara', 'Gördes', 'Kırkağaç', 'Demirci', 'Kula', 'Sarıgöl', 'Saruhanlı', 'Selendi', 'Spil Dağı']

balikesir_ilceler_listesi = ['Balıkesir', 'Merkez', 'Altıeylül', 'Dursunbey', 'Edremit', 'Marmara', 'Sındırgı', 
                             'Bandırma', 'Ayvalık', 'Kepsut', 'Balya', 'Burhaniye', 'Erdek', 'Havran', 'Manyas', 'Savaştepe', 'Susurluk']

# Merge all lists into a single list
bolge_ilceler_listesi = []
bolge_ilceler_listesi.extend(izmir_ilceler_listesi)
bolge_ilceler_listesi.extend(canakkale_ilceler_listesi)
bolge_ilceler_listesi.extend(aydin_ilceler_listesi)
bolge_ilceler_listesi.extend(manisa_ilceler_listesi)
bolge_ilceler_listesi.extend(balikesir_ilceler_listesi)


for ilce in izmir_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_filtered['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = izmir_il_ilceler[izmir_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='red')
        ilce_geometri.plot(ax=ax, color='lightcoral')
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight= 'bold')
        #ax.annotate(text=row['NAME_2'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y + 0.02), xytext=(row['geometry'].centroid.x, row['geometry'].centroid.y + 0.03), color='black', fontsize=10, ha='center')
        
for ilce in izmir_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_eks_yakin['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = izmir_il_ilceler[izmir_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='orange')
        ilce_geometri.plot(ax=ax, color=(1.0, 0.7, 0.4))
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in canakkale_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_filtered['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = canakkale_il_ilceler[canakkale_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='red')
        ilce_geometri.plot(ax=ax, color='lightcoral')
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in canakkale_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_eks_yakin['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = canakkale_il_ilceler[canakkale_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='orange')
        ilce_geometri.plot(ax=ax, color=(1.0, 0.7, 0.4))
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in aydin_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_filtered['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = aydin_il_ilceler[aydin_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='red')
        ilce_geometri.plot(ax=ax, color='lightcoral')
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in aydin_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_eks_yakin['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = aydin_il_ilceler[aydin_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='orange')
        ilce_geometri.plot(ax=ax, color=(1.0, 0.7, 0.4))
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in manisa_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_filtered['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = manisa_il_ilceler[manisa_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='red')
        ilce_geometri.plot(ax=ax, color='lightcoral')
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in manisa_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_eks_yakin['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = manisa_il_ilceler[manisa_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='orange')
        ilce_geometri.plot(ax=ax, color=(1.0, 0.7, 0.4))
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in balikesir_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_filtered['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = balikesir_il_ilceler[balikesir_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='red')
        ilce_geometri.plot(ax=ax, color='lightcoral')
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')

for ilce in balikesir_ilceler_listesi:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce.lower() in df_eks_yakin['Merkez'].str.normalize('NFKD').str.encode('ASCII', 'ignore').str.decode('utf-8').str.lower().values:
        ilce_geometri = balikesir_il_ilceler[balikesir_il_ilceler['NAME_2'] == ilce]
        ilce_geometri.boundary.plot(ax=ax, linewidth=2, color='orange')
        ilce_geometri.plot(ax=ax, color=(1.0, 0.7, 0.4))
        ax.annotate(text=ilce_geometri.iloc[0]['NAME_2'], xy=(ilce_geometri.iloc[0]['geometry'].centroid.x, ilce_geometri.iloc[0]['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight='bold')


# Grafiği gösterme
# plt.show()

sıcaklık_verileri = {}  # Boş bir sözlük oluşturuldu

for index, row in df_2.iterrows():
    ilce = row['Merkezler']  # İlçe adı
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    yillik_sıcaklık = row['Yillik']  # Yıllık ekstrem sıcaklık değeri
    sıcaklık_verileri[normalized_ilce] = yillik_sıcaklık  # Normalized ilçe adı ve sıcaklık değeri sözlüğe eklendi

from matplotlib.widgets import Button
    
# Sıcaklık verilerini çizdirme
ax2 = fig.add_subplot(1, 2, 1)
ax2.axis('off')  # Eksenleri kapatma

# İlçe sıcaklık değerlerini yazdırma
#y_position = 0.9  # Y eksenindeki başlangıç pozisyonu
#for ilce in izmir_il_ilceler['NAME_2']:
#    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
#    if normalized_ilce in sıcaklık_verileri:
#        sıcaklık_değeri = sıcaklık_verileri[normalized_ilce]
#        ax2.annotate(text=f"{ilce}: {sıcaklık_değeri}°C", xy=(0.05, y_position), color='black', fontsize=8, fontweight='bold', ha='left')
#        y_position -= 0.03  # Y ekseninde bir sonraki sıcaklık değeri için pozisyonu güncelleme
        
import mplcursors

# İstasyon noktalarını çizdirme
canakkale_istasyonlar_gdf = gpd.GeoDataFrame(canakkale_istasyonlar, geometry=gpd.points_from_xy(canakkale_istasyonlar['Lon'], canakkale_istasyonlar['Lat']))
canakkale_istasyonlar_gdf.plot(ax=ax, color='white', markersize=5)

izmir_istasyonlar_gdf = gpd.GeoDataFrame(izmir_istasyonlar, geometry=gpd.points_from_xy(izmir_istasyonlar['Lon'], izmir_istasyonlar['Lat']))
izmir_istasyonlar_gdf.plot(ax=ax, color='white', markersize=5)

balikesir_istasyonlar_gdf = gpd.GeoDataFrame(balikesir_istasyonlar, geometry=gpd.points_from_xy(balikesir_istasyonlar['Lon'], balikesir_istasyonlar['Lat']))
balikesir_istasyonlar_gdf.plot(ax=ax, color='white', markersize=5)

manisa_istasyonlar_gdf = gpd.GeoDataFrame(manisa_istasyonlar, geometry=gpd.points_from_xy(manisa_istasyonlar['Lon'], manisa_istasyonlar['Lat']))
manisa_istasyonlar_gdf.plot(ax=ax, color='white', markersize=5)

aydin_istasyonlar_gdf = gpd.GeoDataFrame(aydin_istasyonlar, geometry=gpd.points_from_xy(aydin_istasyonlar['Lon'], aydin_istasyonlar['Lat']))
aydin_istasyonlar_gdf.plot(ax=ax, color='white', markersize=5)

# Noktaların üzerine RAKIM değerini yazdırma

cursor = mplcursors.cursor(ax.scatter(izmir_istasyonlar_gdf.geometry.x, izmir_istasyonlar_gdf.geometry.y, color='white', s=5))
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(f"RAKIM: {izmir_istasyonlar_gdf.iloc[sel.target.index]['RAKIM']}"))

cursor = mplcursors.cursor(ax.scatter(balikesir_istasyonlar_gdf.geometry.x, balikesir_istasyonlar_gdf.geometry.y, color='white', s=5))
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(f"RAKIM: {balikesir_istasyonlar_gdf.iloc[sel.target.index]['RAKIM']}"))

cursor = mplcursors.cursor(ax.scatter(manisa_istasyonlar_gdf.geometry.x, manisa_istasyonlar_gdf.geometry.y, color='white', s=5))
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(f"RAKIM: {manisa_istasyonlar_gdf.iloc[sel.target.index]['RAKIM']}"))

cursor = mplcursors.cursor(ax.scatter(aydin_istasyonlar_gdf.geometry.x, aydin_istasyonlar_gdf.geometry.y, color='white', s=5))
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(f"RAKIM: {aydin_istasyonlar_gdf.iloc[sel.target.index]['RAKIM']}"))

cursor = mplcursors.cursor(ax.scatter(canakkale_istasyonlar_gdf.geometry.x, canakkale_istasyonlar_gdf.geometry.y, color='white', s=5))
cursor.connect(
    "add", lambda sel: sel.annotation.set_text(f"RAKIM: {canakkale_istasyonlar_gdf.iloc[sel.target.index]['RAKIM']}"))

import matplotlib.image as mpimg

# Grafikleri düzenleme
fig.subplots_adjust(left=0.0015, right=0.95, top=0.92, bottom=0.08)  # Kenar boşluklarını ayarlama

fig.savefig(r'D:\data\mgm\extrem.png')

plt.show()


# In[38]:


# 1. Adım: Sütunları çıkart
# df_extreme_deger = df_extreme_deger.drop(columns=['Ekstrem', 'Eks_Yakin'])
# df_eks_yakin_deger = df_eks_yakin_deger.drop(columns=['Ekstrem', 'Eks_Yakin'])


# In[103]:


import pandas as pd
import matplotlib.pyplot as plt

# Veri çerçevelerine yeni isimler atayalım ve ilk sütunları baş harfleri büyük olacak şekilde düzenleyelim
df_extreme_deger.columns = ["Bölge Adı", "Sıcaklık (°C)"]
df_extreme_deger["Bölge Adı"] = df_extreme_deger["Bölge Adı"].apply(lambda x: x.title())

df_eks_yakin_deger.columns = ["Bölge Adı", "Sıcaklık (°C)"]
df_eks_yakin_deger["Bölge Adı"] = df_eks_yakin_deger["Bölge Adı"].apply(lambda x: x.title())

# Tabloyu daha iyi bir görünüme sahip olacak şekilde düzenleme
plt.figure(figsize=(6, 8))

# Açık turuncu rengini RGB 0-1 aralığına getirme
light_orange = (255/255, 165/255, 0)

# Extreme Regions tablosunu çizme
plt.subplot(1, 2, 1)
plt.title("Extreme Regions Table", fontsize=14, fontweight='bold')
plt.table(cellText=df_extreme_deger.values, colLabels=df_extreme_deger.columns, cellLoc='center', loc='center', cellColours=[['lightblue']*len(df_extreme_deger.columns)]*len(df_extreme_deger), colColours=[light_orange]*len(df_extreme_deger.columns))
plt.axis('off')

# Near-Extreme Regions tablosunu çizme
plt.subplot(1, 2, 2)
plt.title("Near-Extreme Regions Table", fontsize=14, fontweight='bold')
plt.table(cellText=df_eks_yakin_deger.values, colLabels=df_eks_yakin_deger.columns, cellLoc='center', loc='center', cellColours=[['lightblue']*len(df_eks_yakin_deger.columns)]*len(df_eks_yakin_deger), colColours=[light_orange]*len(df_eks_yakin_deger.columns))
plt.axis('off')

# Başlık metinlerini tablolardan daha yukarıda konumlandırma
plt.suptitle("Extreme and Near-Extreme Regions Tables", fontsize=16, fontweight='bold', color='red', y=0.98)

# Görseli görüntüleme
plt.tight_layout()

# Tabloyu D:\data\mgm dizinine PNG olarak kaydetme
plt.savefig(r"D:\data\mgm\extreme_and_near_extreme_regions.png", dpi=300)
plt.show()


# In[105]:


from PIL import Image

# İki PNG dosyasını yükleyin
extreme_regions_img = Image.open(r"D:\data\mgm\extreme_and_near_extreme_regions.png")
extrem_img = Image.open(r"D:\data\mgm\extrem.png")

# İki resmin boyutunu alın
extreme_regions_width, extreme_regions_height = extreme_regions_img.size
extrem_width, extrem_height = extrem_img.size

# Yeni bir resim oluşturun, iki resmi yatayda birleştirmek için genişlikleri toplayın
combined_width = extreme_regions_width + extrem_width
combined_height = max(extreme_regions_height, extrem_height)

combined_img = Image.new('RGB', (combined_width, combined_height))

# İlk resmi (extreme_regions_img) sol tarafa yerleştirin
combined_img.paste(extreme_regions_img, (0, 0))

# İkinci resmi (extrem_img) sağ tarafa yerleştirin
combined_img.paste(extrem_img, (extreme_regions_width, 0))

# Birleştirilmiş resmi kaydedin
combined_img.save(r"D:\data\mgm\combined.png")

display(combined_img)


# In[34]:


import unicodedata
from IPython.display import Image
fig.savefig(r'D:\data\mgm\extrem.png')

# Görüntüyü başka bir sekmede gösterme
Image(filename=r'D:\data\mgm\extrem.png')

# import pandas as pd

# Veriyi okuma
df = pd.read_csv(r"D:\data\mgm\12.07.2023.csv")

# "Max1" ile "Max5" sütunlarını birleştirin
all_max_values = pd.concat([df["Max1"], df["Max2"], df["Max3"], df["Max4"], df["Max5"]])

# Yeni DataFrame oluşturma
data_frame = pd.DataFrame({"Max": all_max_values})

# Yeniden indeksleme
data_frame = data_frame.reset_index(drop=True)

# Örnek olarak veriyi gösterme
print(data_frame)

# Yeni DataFrame'i CSV dosyasına kaydetme
data_frame.to_csv(r"D:\data\mgm\data_frame.csv", index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veriyi okuma
df = pd.read_csv(r"D:\data\mgm\12.07.2023.csv")

# Maksimum sıcaklık değerlerini birleştirin
all_max_values = pd.concat([df["Max1"], df["Max2"], df["Max3"], df["Max4"], df["Max5"]])

# Z-Skoru hesaplama
z_scores = np.abs((all_max_values - all_max_values.mean()) / all_max_values.std())

# Aykırı değerleri belirleme (Z-Skoru 3'ten büyük veya -3'ten küçük olanlar aykırı kabul edilir)
outliers = all_max_values[z_scores > 3]

# Aykırı olmayan değerleri belirleme
filtered_values = all_max_values[z_scores <= 3]

# Aykırı değerlerle birlikte yeni DataFrame oluşturma
filtered_df = pd.DataFrame({"Max": filtered_values})

# Örnek olarak aykırı değerleri gösterme
print("Aykırı Değerler:")
print(outliers)

# Aykırı olmayan değerleri gösterme
print("\nAykırı Olmayan Değerler:")
print(filtered_df)

# CSV dosyasına kaydetme
filtered_df.to_csv(r"D:\data\mgm\filtered_data.csv", index=False)import pandas as pd
import matplotlib.pyplot as plt

# Veriyi okuma
filtered_df = pd.read_csv(r"D:\data\mgm\filtered_data.csv")

# Zaman serisi çizimi
plt.plot(filtered_df["Max"])
plt.xlabel("Tarih")
plt.ylabel("Maksimum Sıcaklık")
plt.title("Maksimum Sıcaklık Zaman Serisi (Aykırı Değerler Hariç)")
plt.show()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Veriyi okuma
filtered_df = pd.read_csv(r"D:\data\mgm\filtered_data.csv")

# Bağımsız değişken (X) ve bağımlı değişken (y) olarak verileri ayırma
X = np.arange(len(filtered_df)).reshape(-1, 1)
y = filtered_df["Max"]

# Verileri eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Doğrusal regresyon modelini oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Modeli kullanarak tahmin yapma
y_pred = model.predict(X_test)

# Tahminleri ve gerçek değerleri görselleştirme
plt.plot(X_test, y_pred, label="Tahmin", color='red')
plt.plot(X_test, y_test, label="Gerçek")
plt.xlabel("Veri Örnekleri")
plt.ylabel("Maksimum Sıcaklık")
plt.title("Doğrusal Regresyon ile Maksimum Sıcaklık Tahmini")
plt.legend()
plt.show()import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Veriyi okuma
filtered_df = pd.read_csv(r"D:\data\mgm\filtered_data.csv")

# Bağımsız değişken (X) ve bağımlı değişken (y) olarak verileri ayırma
X = np.arange(len(filtered_df)).reshape(-1, 1)
y = filtered_df["Max"]

# Modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X, y)

# Gelecek 5 gün için tahmin yapma
gelecek_gunler = np.arange(len(filtered_df), len(filtered_df) + 365).reshape(-1, 1)
tahminler = model.predict(gelecek_gunler)

# Tahminleri ve gerçek değerleri görselleştirme
plt.plot(X, y, label="Eğitim Verileri")
plt.plot(gelecek_gunler, tahminler, label="Gelecek Tahminleri", color='red')
plt.xlabel("Veri Örnekleri")
plt.ylabel("Maksimum Sıcaklık")
plt.title("Doğrusal Regresyon ile Gelecek 365 Gün Maksimum Sıcaklık Tahmini")
plt.legend()
plt.show()# İlk satırın değerini saklamak için boş bir değişken tanımlayın
ilk_satir_degeri = None

# df_filtered veri setinde her bir satır için döngü oluşturma
for idx, row in df_filtered.iterrows():
    merkez = row['Merkez']
    # İlk satırın değerini saklama
    if ilk_satir_degeri is None:
        ilk_satir_degeri = merkez
    print(merkez)

# Döngüden çıktıktan sonra ilk satırın değerini yazdırma
print("İlk satırın değeri:", ilk_satir_degeri)min3_degeri = tum_degerler['selcuk']['Min3']
print(f"{'selcuk'} bölgesinin Min3 değeri: {min3_degeri}")import webbrowserimport pandas as pd
from bs4 import BeautifulSoup

file_path = r"D:\data\mgm\test.html"  # HTML dosya yolunu belirtin

# HTML dosyasını açın ve içeriği okuyun
with open(file_path, 'r') as file:
    html_content = file.read()

# BeautifulSoup ile HTML içeriğini analiz edin
soup = BeautifulSoup(html_content, 'html.parser')

# Tablo etiketlerini bulun
tables = soup.find_all('table')

# Tüm tabloları bir DataFrame olarak dönüştürün
dfs = []
for table in tables:
    df = pd.read_html(str(table))[0]
    dfs.append(df)
    
# DataFrame'leri birleştirin
combined_df = pd.concat(dfs)

# CSV olarak kaydedin
output_csv = r"D:\data\mgm\output.csv"  # Çıktı CSV dosya yolunu belirtin
combined_df.to_csv(output_csv, index=False)

print("HTML içeriği başarıyla CSV dosyasına kaydedildi.")