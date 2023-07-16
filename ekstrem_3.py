#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install geopandas


# In[ ]:


pip install mplcursors


# In[13]:


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[29]:


# Excel dosyasını oku
# CSV dosyasını okuma
df_2 = pd.read_csv("D:\data\mgm\İzmir_Ekstrem.csv")
df = pd.read_csv((r"D:\data\mgm\12.07.2023.csv"))


# In[30]:


df_2


# In[31]:


guzelbahce_satir = df.loc[df['Merkezler'] == 'GUZELBAHCE']
max_sicaklik = guzelbahce_satir[['Max1', 'Max2', 'Max3', 'Max4', 'Max5']].max().max()
ekstrem_sicaklik = guzelbahce_satir['14.07.23'].values[0]
print(f"Guzelbahce için maksimum sıcaklık: {max_sicaklik}°C")
print(f"Guzelbahce için ekstrem sıcaklık: {ekstrem_sicaklik}°C")


# In[34]:


veri_cerceveleri = {}  # Veri çerçevelerini saklamak için boş bir sözlük oluştur

sütun_adi = 'Merkezler'  # İlgili sütun adını belirtin

for kelime in df[sütun_adi].unique():
    kelime = kelime.lower()  # Kelimeyi küçük harflere çevir
    veri_cerceveleri[kelime] = df[df[sütun_adi].str.lower() == kelime]


# In[35]:


veri_cerceveleri_2 = {}  # İkinci veri çerçevelerini saklamak için boş bir sözlük oluştur

sütun_adı_2 = 'Merkezler'  # İkinci veri setindeki ilgili sütun adını belirtin

for kelime_2 in df_2[sütun_adı_2].unique():
    kelime_2 = kelime_2.lower()  # Kelimeyi küçük harflere çevir
    kelime_2 = kelime_2.replace(" ", "_")  # Boşlukları alt çizgi ile değiştir
    veri_cerceveleri_2[kelime_2] = df_2[df_2[sütun_adı_2].str.lower() == kelime_2]

# Örnek kullanım: bornova = veri_cerceveleri_2['bornova']


# In[36]:


tum_degerler = {}  # Tüm satırlara ait değerleri saklamak için boş bir sözlük oluştur

for kelime, veri_cercevesi in veri_cerceveleri.items():
    degerler = {}
    for sütun in veri_cercevesi.columns[1:]:  # İlk sütunu hariç al
        degerler[sütun] = veri_cercevesi[sütun].values[0]
    tum_degerler[kelime] = degerler

# Örnek kullanım: tum_degerler['cigli'] ile Cigli'ye ait bütün sütun değerlerine erişebilirsiniz


# In[37]:


tum_degerler_2 = {}  # Tüm satırlara ait değerleri saklamak için boş bir sözlük oluştur

for kelime_2, veri_cercevesi_2 in veri_cerceveleri_2.items():
    degerler_2 = {}
    if not veri_cercevesi_2.empty:  # Veri çerçevesi boş değilse işlemleri yap
        for sütun_2 in veri_cercevesi_2.columns[1:]:  # İlk sütunu hariç al
            degerler_2[sütun_2] = veri_cercevesi_2[sütun_2].values[0]
    tum_degerler_2[kelime_2] = degerler_2

# Örnek kullanım: tum_degerler_2['bornova'] ile Bornova'ya ait bütün sütun değerlerine erişebilirsiniz


# In[38]:


tum_degerler_2['selcuk']


# In[39]:


tum_degerler['edremit']
#tum_degerler['edremit']


# In[ ]:





# In[40]:


extreme_merkezler = []
data = []

for kelime, degerler in tum_degerler.items():
    if kelime in tum_degerler_2:
        yillik_sicaklik = tum_degerler_2[kelime]['Yıllık']
        max_sicaklikler = [degerler[f'Max{i}'] for i in range(1, 6)]

        max_sicaklik = max(max_sicaklikler)
        print('Güncel Sıcaklık: ', max_sicaklik)
        print('Eski Ekstrem Sıcaklık: ', yillik_sicaklik)

        if max_sicaklik > yillik_sicaklik:
            print(f"{kelime} bölgesinde ekstrem sıcaklık değerleri tespit edildi.")
            extreme_merkezler.append(kelime)
            data.append({'Merkez': kelime, 'Ekstrem': True, 'Eks_Yakin': False})
        elif abs(max_sicaklik - yillik_sicaklik) < 2:
            print(f"{kelime} bölgesinde ekstrem sıcaklık değerlerine yakın bir durum tespit edildi.")
            extreme_merkezler.append(kelime)
            data.append({'Merkez': kelime, 'Ekstrem': False, 'Eks_Yakin': True})
        else:
            print(f"{kelime} bölgesinde ekstrem sıcaklık değerleri tespit edilmedi.")
            data.append({'Merkez': kelime, 'Ekstrem': False, 'Eks_Yakin': False})
    else:
        # print(f"{kelime} bölgesine ait yıllık sıcaklık verisi bulunamadı.")
        data.append({'Merkez': kelime, 'Ekstrem': False, 'Eks_Yakin': False})



# extreme_merkezler listesindeki ekstrem merkezlerin isimleri
for merkez in extreme_merkezler:
    print(f"Ekstrem olarak tespit edilen merkez: {merkez}")

# Data array'ini görüntüleme
for d in data:
    print(f"Merkez: {d['Merkez']}, Ekstrem: {d['Ekstrem']}")
    
data_dict = {
    'Merkez': [d['Merkez'] for d in data],
    'Ekstrem': [d['Ekstrem'] for d in data]
}

df_ekstrem = pd.DataFrame(data_dict)

print(df_ekstrem)


# In[41]:


data_filtered = [d for d in data if d['Ekstrem']]

data_dict_filtered = {
    'Merkez': [d['Merkez'] for d in data_filtered],
    'Ekstrem': [d['Ekstrem'] for d in data_filtered]
}

df_filtered = pd.DataFrame(data_dict_filtered)

print(df_filtered)


# In[42]:


import pandas as pd

eks_yakin_data = [{'Merkez': 'guzelbahce', 'Ekstrem': False, 'Eks_Yakin': True},
                  {'Merkez': 'kiraz', 'Ekstrem': False, 'Eks_Yakin': True}]

df_eks_yakin = pd.DataFrame(eks_yakin_data)
print(df_eks_yakin)


# In[43]:


import re

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
    
    plt.plot(tarihler_xticks, min_sıcaklık_verileri, marker='o', linestyle='-', color='b', label='Min Sıcaklık')
    plt.plot(tarihler_xticks, max_sıcaklık_verileri, marker='o', linestyle='-', color='g', label='Max Sıcaklık')
    plt.title(f'Ekstrem Değer ile {center_name.capitalize()} Sıcaklık Grafiği')
    plt.xlabel('Tarih')
    plt.ylabel('Sıcaklık')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Ekstrem değeri göstermek için bir çizgi çizin
    plt.axhline(y=extreme_value, color='r', linestyle='--', linewidth=2, label='Ekstrem Değer')
    y_min = min(min_sıcaklık_verileri + [extreme_value]) - 1
    y_max = max(max_sıcaklık_verileri + [extreme_value]) + 1
    plt.ylim(y_min, y_max)  # Y eksenini minimum ve maksimum sıcaklık değerleri arasında ayarlayın

    # Etiketleri noktanın üzerine yerleştirin
    for x, y_max, y_min in zip(tarihler_xticks, max_sıcaklık_verileri, min_sıcaklık_verileri):
        plt.text(x, y_max, str(y_max), ha='center', va='bottom', color='white', backgroundcolor='black')
        plt.text(x, y_min, str(y_min), ha='center', va='top', color='white', backgroundcolor='black')
    
    # Ekstrem değeri yazdır
    plt.text(tarihler_xticks[-1], extreme_value, f'Ekstrem: {extreme_value}', ha='right', va='bottom',
             color='black', backgroundcolor='white')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Örnek kullanım: aliaga için grafiği oluştur
center_name = 'aliaga'
center_data = veri_cerceveleri[center_name]
extreme_value = tum_degerler_2[center_name]['Yıllık']

plot_with_extreme(center_name, center_data, extreme_value)


# In[44]:


# CSV dosyasını okuma
df = pd.read_csv(r"D:\data\mgm\istasyonlar.csv")

# Veriyi görüntüleme
# print(df)


# In[45]:


# Çanakkale bölgesinin lat ve lon sınırları
min_lat = 39.600
max_lat = 40.400
min_lon = 26.000
max_lon = 27.400

# Çanakkale bölgesi içindeki istasyonları seçme
canakkale_istasyonlar = df[(df['Lat'] >= min_lat) & (df['Lat'] <= max_lat) & (df['Lon'] >= min_lon) & (df['Lon'] <= max_lon)]

# Seçilen istasyonları görüntüleme
# print(canakkale_istasyonlar)


# In[46]:


# Bornova ilçesinin adını alın
bornova_merkez = df_filtered.loc[df_filtered['Merkez'] == 'bornova', 'Merkez'].iloc[0]

# print(bornova_merkez)


# In[47]:


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
tum_ilceler = izmir_il_ilceler.append([manisa_il_ilceler, canakkale_il_ilceler, aydin_il_ilceler, balikesir_il_ilceler])

# Haritayı çizdirme
fig, ax = plt.subplots(figsize=(20, 20))
izmir_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='blue')
manisa_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='green')
canakkale_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='red')
aydin_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='purple')
balikesir_il_sınırlar.boundary.plot(ax=ax, linewidth=2, color='orange')

# Tüm ilçeleri çizdirme ve rengini değiştirme
tum_ilceler.boundary.plot(ax=ax, linewidth=0.5, color='black')
tum_ilceler.plot(ax=ax, color='lightblue')

# İlçe isimlerini yazdırma
for idx, row in tum_ilceler.iterrows():
    ax.annotate(text=row['NAME_2'], xy=(row['geometry'].centroid.x, row['geometry'].centroid.y), color='black', fontsize=8, ha='center', fontweight= 'bold')

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

import unicodedata

izmir_ilceler_listesi = ['Aliağa', 'Balçova', 'Bayındır', 'Bayraklı', 'Bergama', 'Beydağ', 'Bornova',
                        'Buca', 'Çeşme', 'Çiğli', 'Dikili', 'Foça', 'Gaziemir', 'Güzelbahçe', 'Karabağlar',
                        'Karaburun', 'Karşıyaka', 'Kemalpaşa', 'Kınık', 'Kiraz', 'Konak', 'Menderes',
                        'Menemen', 'Narlıdere', 'Ödemiş', 'Seferihisar', 'Selçuk', 'Tire', 'Torbalı', 'Urla', 'Bayındır']

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



# Grafiği gösterme
# plt.show()

sıcaklık_verileri = {}  # Boş bir sözlük oluşturuldu

for index, row in df_2.iterrows():
    ilce = row['Merkezler']  # İlçe adı
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    yillik_sıcaklık = row['Yıllık']  # Yıllık ekstrem sıcaklık değeri
    sıcaklık_verileri[normalized_ilce] = yillik_sıcaklık  # Normalized ilçe adı ve sıcaklık değeri sözlüğe eklendi

from matplotlib.widgets import Button
    
# Sıcaklık verilerini çizdirme
ax2 = fig.add_subplot(1, 2, 1)
ax2.axis('off')  # Eksenleri kapatma

# İlçe sıcaklık değerlerini yazdırma
y_position = 0.9  # Y eksenindeki başlangıç pozisyonu
for ilce in izmir_il_ilceler['NAME_2']:
    normalized_ilce = unicodedata.normalize('NFKD', ilce).encode('ASCII', 'ignore').decode('utf-8')
    if normalized_ilce in sıcaklık_verileri:
        sıcaklık_değeri = sıcaklık_verileri[normalized_ilce]
        ax2.annotate(text=f"{ilce}: {sıcaklık_değeri}°C", xy=(0.05, y_position), color='black', fontsize=8, fontweight='bold', ha='left')
        y_position -= 0.03  # Y ekseninde bir sonraki sıcaklık değeri için pozisyonu güncelleme

# Grafikleri düzenleme
fig.subplots_adjust(left=0.0015, right=0.95, top=0.92, bottom=0.08)  # Kenar boşluklarını ayarlama

fig.savefig(r'D:\data\mgm\extrem.png')

plt.show()


# In[68]:


pip install Flask


# In[70]:


from flask import Flask, render_template

# Flask uygulamasını oluşturma
app = Flask(__name__)

# İstasyon verilerini işleme, haritayı oluşturma ve kaydetme işlemleri burada olacak
# İstasyon verilerini, sıcaklık değerlerini ve harita işlemlerini bu fonksiyon içinde yapabilirsiniz.
# Son olarak oluşturulan "extrem.png" dosyasını bir şekilde kullanılabilir bir yere kaydedebilirsiniz.

# Ana sayfa için route tanımlama
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()


# In[64]:


import pandas as pd

# Veriyi okuma
df = pd.read_csv(r"D:\data\mgm\12.07.2023.csv")

# "Max1" sütunundaki değerleri alın
max1_values = df["Max1"]

# "Max2", "Max3", "Max4", "Max5" sütunundaki değerleri alın
max2_values = df["Max2"]
max3_values = df["Max3"]
max4_values = df["Max4"]
max5_values = df["Max5"]

# "Max1" ile "Max5" sütunlarını birleştirin
all_max_values = pd.concat([max1_values, max2_values, max3_values, max4_values, max5_values])

# Yeni DataFrame oluşturma
data_frame = pd.DataFrame({"Max": all_max_values})

# Örnek olarak veriyi gösterme
print(data_frame)
# Yeni DataFrame'i CSV dosyasına kaydetme
data_frame.to_csv(r"D:\data\mgm\data_frame.csv", index=False)


# In[67]:


import pandas as pd
from datetime import timedelta

# Veriyi okuma
df = pd.read_csv(r"D:\data\mgm\12.07.2023.csv")

# Verilerin başlangıç tarihini belirleyin
start_date = pd.to_datetime('2023-07-12')  # Örnek olarak başlangıç tarihini belirleyin

# "Max1" sütunundaki değerleri alın
max1_values = df["Max1"]

# ARIMA modelini oluşturun ve eğitin
model = ARIMA(max1_values, order=(5, 1, 0))  # Örnek olarak (p, d, q) = (5, 1, 0)
model_fit = model.fit()

# Gelecekteki değerleri tahmin etmek için modeli kullanın
n_forecast = 7  # Örnek olarak 7 gün sonrasını tahmin ediyoruz
forecast = model_fit.forecast(steps=n_forecast)

# Yeni tarihleri oluşturun
new_dates = [start_date + timedelta(days=i) for i in range(1, n_forecast+1)]

# Yeni veri setini oluşturun
new_data = pd.DataFrame({"Tarih": new_dates, "Max": forecast})

# Orijinal veri ile yeni veriyi birleştirin
extended_data = pd.concat([df, new_data], ignore_index=True)

# Sonuçları gösterin
print(extended_data)

# Yeni veriyi CSV dosyasına kaydedin
extended_data.to_csv(r"D:\data\mgm\extended_data.csv", index=False)


# In[48]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Verilerinizi yükleyin veya hazırlayın
# Verileri df_filtered'dan alıyorum. 'Ekstrem' sütunu hedef değişkenimiz (target).
X = df_filtered[['Sıcaklık_Sütunu1', 'Sıcaklık_Sütunu2', ...]]  # Sıcaklık sütunlarını veri kümenize göre değiştirin
y = df_filtered['Ekstrem']

# Verileri eğitim ve test kümesi olarak ayırın (örn. %80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression modelini oluşturun ve eğitin
model = LogisticRegression()
model.fit(X_train, y_train)

# Test veri kümesi üzerinde modelin performansını değerlendirin
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report_result)
print("Confusion Matrix:")
print(confusion_mat)


# In[105]:


# İlk satırın değerini saklamak için boş bir değişken tanımlayın
ilk_satir_degeri = None

# df_filtered veri setinde her bir satır için döngü oluşturma
for idx, row in df_filtered.iterrows():
    merkez = row['Merkez']
    # İlk satırın değerini saklama
    if ilk_satir_degeri is None:
        ilk_satir_degeri = merkez
    print(merkez)

# Döngüden çıktıktan sonra ilk satırın değerini yazdırma
print("İlk satırın değeri:", ilk_satir_degeri)


# In[106]:


min3_degeri = tum_degerler['selcuk']['Min3']
print(f"{'selcuk'} bölgesinin Min3 değeri: {min3_degeri}")


# In[ ]:


import webbrowser


# In[ ]:


import pandas as pd
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

