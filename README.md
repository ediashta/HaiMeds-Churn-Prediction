[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/vP9K9cz6)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11486151&assignment_repo_type=AssignmentRepo)
# Phase 2 - Milestones 1

_Milestones ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Full Time Program Phase 2 khususnya pada Artificial Neural Network._

---

## Assignment Objectives

*Milestones 1* ini dibuat guna mengevaluasi Pembelajaran Phase 2 :

- Mampu memahami konsep Artificial Neural Network.
- Mampu mempersiapkan data untuk digunakan dalam model Artificial Neural Network.
- Mampu mengimplementasikan Artificial Neural Network dengan data yang diberikan.
- Mampu menganalisis dan menjelaskan layer yang dibuat.

---

## Problems

Sebuah perusahaan ingin meminimalisir resiko seorang customer untuk berhenti memakai produk yang mereka tawarkan. Bantulah perusahaan tersebut untuk memprediksi customer yang akan berhenti (churn) dari dataset yang diberikan. 

*Dataset terlampir pada repository*

---

## Dataset Description

Dataset name : `churn.csv`

| Column | Description |
| --- | --- |
| `user_id` | ID of a customer |
| `age` | Age of a customer |
| `gender` | Gender of a customer |
| `region_category` | Region that a customer belongs to |
| `membership_category` | Category of the membership that a customer is using |
| `joining_date` | Date when a customer became a member |
| `joined_through referral` | Whether a customer joined using any referral code or ID |
| `preferred_offer types` | Type of offer that a customer prefers |
| `medium_of operation` | Medium of operation that a customer uses for transactions |
| `internet_option` | Type of internet service a customer uses |
| `last_visit_time` | The last time a customer visited the website |
| `days_since_last_login` | Number of days since a customer last logged into the website |
| `avg_time_spent` | Average time spent by a customer on the website |
| `avg_transaction_value` | Average transaction value of a customer |
| `avg_frequency_login_days` | Number of times a customer has logged in to the website |
| `points_in_wallet` | Points awarded to a customer on each transaction |
| `used_special_discount` | Whether a customer uses special discounts offered |
| `offer_application_preference` | Whether a customer prefers offers |
| `past_complaint` | Whether a customer has raised any complaints |
| `complaint_status` | Whether the complaints raised by a customer was resolved |
| `feedback` | Feedback provided by a customer |
| `churn_risk_score` | Churn score <br><br> `0` : Not churn <br> `1` : Churn |

---

## Assignment Instructions

*Milestones 1* dikerjakan dalam format *notebook* dan *model deployment* dengan beberapa *kriteria wajib* di bawah ini:

1. Deep Learning framework yang digunakan adalah *TensorFlow*.

2. Ada penggunaan library visualisasi, seperti *matplotlib*, *seaborn*, atau yang lain.

3. Isi *notebook* harus mengikuti *outline* di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, gambaran besar dataset yang digunakan, dan *objective* yang ingin dicapai.
   
   2. Import Libraries
      > *Cell* pertama pada *notebook* **harus berisi dan hanya berisi** semua *library* yang digunakan dalam *project*.
   
   3. Data Loading
      > Bagian ini berisi proses penyiapan data sebelum dilakukan eksplorasi data lebih lanjut. Proses Data Loading dapat berupa memberi nama baru untuk setiap kolom, mengecek ukuran dataset, dll.
   
   4. Exploratory Data Analysis (EDA)
      > Bagian ini berisi explorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.

   5. Feature Engineering
      > Bagian ini berisi proses penyiapan data untuk proses pelatihan model, seperti pembagian data menjadi train-val-test, transformasi data (normalisasi, encoding, dll.), dan proses-proses lain yang dibutuhkan.   
   
   6. Model Definition
      > Bagian ini berisi cell untuk mendefinisikan model. Jelaskan alasan menggunakan suatu algoritma/model, hyperparameter yang dipakai, jenis penggunaan metrics yang dipakai, dan hal lain yang terkait dengan model.

   7. Model Training
      > Cell pada bagian ini hanya berisi code untuk melatih model dan output yang dihasilkan. Lakukan beberapa kali proses training dengan hyperparameter yang berbeda untuk melihat hasil yang didapatkan. Analisis dan narasikan hasil ini pada bagian Model Evaluation.
   
   8. Model Evaluation
      > Pada bagian ini, dilakukan evaluasi model yang harus menunjukkan bagaimana performa model berdasarkan metrics yang dipilih. Hal ini harus dibuktikan dengan visualisasi tren performa dan/atau tingkat kesalahan model. **Lakukan analisis terkait dengan hasil pada model dan tuliskan hasil analisisnya**.

   9. Model Saving
      > Pada bagian ini, dilakukan proses penyimpanan model dan file-file lain yang terkait dengan hasil proses pembuatan model. Pilihlah 1 arsitektur ANN yang terbaik berdasarkan hasil evaluasi sebelumnya.
   
   10. Model Inference
       > Model yang sudah dilatih akan dicoba pada data yang bukan termasuk ke dalam train-set, val-set, ataupun test-set. Data ini harus dalam format yang asli, bukan data yang sudah di-scaled.
   
   11. Pengambilan Kesimpulan
       > Pada bagian terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan *objective* yang sudah ditulis di bagian pengenalan.

5. *Notebook* harus diupload dalam akun GitHub Classroom masing-masing siswa untuk selanjutnya dinilai.

6. Presentasikan model yang telah dibuat pada P2W2D3PM.

---

## Assignment Submission

- Simpan assignment pada sesi ini dengan nama `h8dsft_P2M1_<nama-student>.ipynb`, misal `h8dsft_P2M1_raka_ardhi.ipynb`.

- Push Assigment yang telah Anda buat ke akun Github Classroom masing-masing.

- Untuk Model Deployment :
  * Buat sebuah folder bernama `deployment` dan masukkan semua file yang berkaitan dengan deployment ke folder ini.
  * Buat sebuah file bernama `url.txt` yang berisi URL deployment.
  * Contoh bentuk isi repository dengan Model Deployment.
    ```
    ├── deployment/
    │   ├── app.py
    │   ├── eda.py
    │   ├── prediction.py
    │   ├── model.h5
    │   ├── requirements.txt
    ├── h8dsft_P2M1_raka_ardhi.ipynb
    ├── h8dsft_P2M1_raka_ardhi_inference.ipynb
    ├── churn.csv
    ├── url.txt
    └── README.md
    ```
---

## Assignment Rubrics

### Code Review

| Criteria|Meet Expectations|Points|
| --- | --- | --- |
| Feature Engineering | Mampu melakukan preprocessing dataset sebelum melakukan proses modeling (split data, normalisasi, encoding, dll) | 35 pts |
| TensorFlow Pipeline | Mengimplementasikan TensorFlow Pipeline pada domain kasus yang dipilih. **Pipeline yang dimaksud adalah Loading and Preprocessing Data dengan menggunakan modul TensorFlow** | 5 pts |
| ANN Training | Membuat model Artifical Neural Network dengan menggunakan Sequential API dan Functional API  | 10 pts (untuk Sequential API), <br><br> 10 pts (untuk Functional API) |
| ANN Improvement | Membuat improvisasi model Artificial Neural Network dengan menggunakan Sequential API dan Functional API | 13 pts (untuk Sequential API), <br><br> 13 pts (untuk Functional API) |
| Model Inference | Mencoba model yang telah dibuat dengan data baru | 10 pts |
| Runs Perfectly | Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar | 10 pts |

```
Catatan mengenai ANN Training : 

1. Narasikan mengenai layer yang Anda buat beserta hyperparameter yang Anda tulis.
2. Sebaiknya lakukan training dengan lebih dari 20 epochs agar dapat terlihat performa model dengan lebih jelas.
3. Buatlah arsitektur yang berbeda antara Sequential API dan Functional API.

Catatan mengenai ANN Improvement :
1. Narasikan mengenai apa yang Anda lakukan untuk memperbaiki performansi dari model-model yang tercipta pada ANN Training.
2. Sama seperti ANN Training, Anda diminta untuk membuat arsitektur Sequential API dan Functional API yang berbeda satu sama lain.
3. Arsitektur ANN Sequential API yang dibuat untuk rubric ANN Improvement akan dibandingkan dengan aristektur ANN Sequential API dari rubric ANN Training dan harus memiliki performansi yang lebih baik, begitu juga untuk Functional API.
4. Contoh definisi performansi yang lebih baik : memiliki nilai accuracy atau metrics lain yang lebih baik, proses training lebih cepat meski nilai accuracy sama, nilai accuracy sama namun ukuran model lebih kecil, dll.
5. Model yang dipilih sebagai model terbaik haruslah model dari ANN Improvement, tidak boleh berasal dari ANN Training. Model terbaik ini akan digunakan pada Model Inference dan Model Deployment.
6. Anda tidak diperkenankan menggunakan transfer learning. Ujilah beberapa arsitektur buatan Anda sendiri.
7. Anda dipersilakan menggunakan TensorFlow Callback.
```

### Readability

| Criteria | Meet Expectations | Points|
| --- | --- | --- |
| Tertata Dengan Baik | Semua baris kode terdokumentasi dengan baik dengan menggunakan Markdown untuk penjelasan kode. | 15 pts |

```
Kriteria tertata dengan baik diantaranya adalah : 

1. Terdapat section Perkenalan yang jelas
2. Tidak menyalin markdown dari tugas lain.
3. Import library rapih (terdapat dalam 1 cell dan tidak ada unused libs).
4. Pemakaian fungsi markdown yang optimal (Heading, text formating, dll). 
5. Terdapat komentar pada setiap baris kode.
6. Adanya pemisah yang jelas antar section, dll.
7. Tidak adanya typo.
```

### Analysis

| Criteria | Meet Expectations | Points|
| --- |--- |--- |
| Model Analysis | Menganalisa informasi dari model yang telah dibuat | 35 pts |
| Overall Analysis | Menarik informasi/kesimpulan dari keseluruhan kegiatan yang dilakukan | 20 pts |

```
Contoh kriteria analisa yang baik diantaranya adalah: 

1. Terdapat penjelasan macam-macam hasil metric evaluasi dan interpretasinya terhadap kasus yang diselesaikan.
2. Dapat menjelaskan kelemahan/kekurangan dan kelebihan dari model yang dibuat.
3. Dapat memberikan statement untuk improvement selanjutnya dari model yang dibuat. 
4. Sebutkan insight yang dapat diambil setelah proses EDA, dll.
```

### Deployment

| Criteria | Meet Expectations | Points |
| --- | --- | --- |
| Model Deployment | Membuat webapps terhadap project yang telah dibuat. | 15 pts |

```
Catatan mengenai Model Deployment : 

1. Ketiadaan URL deployment ataupun source code deployment di repository, akan tetap diperhitungkan untuk menilai bagian Model Deployment. 
2. Tidak diperkenankan adanya informasi tambahan/informasi susulan seperti lupa memberikan URL deployment atau lupa mengupload source code via apapun (DM buddy, email, atau yang lain).
3. Student akan dianggap tidak melakukan Model Deployment jika tidak ada URL deployment dan source code deployment di repository.
```

---

```
Total Points : 191
```

---

## Notes

- **Deadline : P2W1D5 pukul 18:00 WIB.**

- **Keterlambatan pengumpulan tugas mengakibatkan skor Milestone 1 menjadi 0.**
