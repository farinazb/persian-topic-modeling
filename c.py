from __future__ import unicode_literals
from hazm import *
import numpy as np
import pandas as pd
import re
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

stopwords = set(
    """
و
به
در
از
که
این
را
با
است
برای
آن
یک
خود
تا
کرد
بر
هم
نیز
گفت
می‌شود
وی
شد
دارد
ما
اما
یا
شده
باید
هر
آنها
بود
او
دیگر
دو
مورد
می‌کند
شود
کند
وجود
بین
پیش
شده‌است
پس
نظر
اگر
همه
یکی
حال
هستند
من
کنند
نیست
باشد
چه
بی
می
بخش
می‌کنند
همین
افزود
هایی
دارند
راه
همچنین
روی
داد
بیشتر
بسیار
سه
داشت
چند
سوی
تنها
هیچ
میان
اینکه
شدن
بعد
جدید
ولی
حتی
کردن
برخی
کردند
می‌دهد
اول
نه
کرده‌است
نسبت
بیش
شما
چنین
طور
افراد
تمام
درباره
بار
بسیاری
می‌تواند
کرده
چون
ندارد
دوم
بزرگ
طی
حدود
همان
بدون
البته
آنان
می‌گوید
دیگری
خواهد‌شد
کنیم
قابل
یعنی
رشد
می‌توان
وارد
کل
ویژه
قبل
براساس
نیاز
گذاری
هنوز
لازم
سازی
بوده‌است
چرا
می‌شوند
وقتی
گرفت
کم
جای
حالی
تغییر
پیدا
اکنون
تحت
باعث
مدت
فقط
زیادی
تعداد
آیا
بیان
رو
شدند
عدم
کرده‌اند
بودن
نوع
بلکه
جاری
دهد
برابر
مهم
بوده
اخیر
مربوط
امر
زیر
گیری
شاید
خصوص
آقای
اثر
کننده
بودند
فکر
کنار
اولین
سوم
سایر
کنید
ضمن
مانند
باز
می‌گیرد
ممکن
حل
دارای
پی
مثل
می‌رسد
اجرا
دور
منظور
کسی
موجب
طول
امکان
آنچه
تعیین
گفته
شوند
جمع
خیلی
علاوه
گونه
تاکنون
رسید
ساله
گرفته
شده‌اند
علت
چهار
داشته‌باشد
خواهد‌بود
طرف
تهیه
تبدیل
مناسب
زیرا
مشخص
می‌توانند
نزدیک
جریان
روند
بنابراین
می‌دهند
یافت
نخستین
بالا
پنج
ریزی
عالی
چیزی
نخست
بیشتری
ترتیب
شده‌بود
خاص
خوبی
خوب
شروع
فرد
کامل
غیر
می‌رود
دهند
آخرین
دادن
جدی
بهترین
شامل
گیرد
بخشی
باشند
تمامی
بهتر
داده‌است
حد
نبود
کسانی
می‌کرد
داریم
علیه
می‌باشد
دانست
ناشی
داشتند
دهه
می‌شد
ایشان
آنجا
گرفته‌است
دچار
می‌آید
لحاظ
آنکه
داده
بعضی
هستیم
اند
برداری
نباید
می‌کنیم
نشست
سهم
همیشه
آمد
اش
وگو
می‌کنم
حداقل
طبق
جا
خواهد‌کرد
نوعی
چگونه
رفت
هنگام
فوق
روش
ندارند
سعی
بندی
شمار
کلی
کافی
مواجه
همچنان
زیاد
سمت
کوچک
داشته‌است
چیز
پشت
آورد
حالا
روبه
سال‌های
دادند
می‌کردند
عهده
نیمه
جایی
دیگران
سی
بروز
یکدیگر
آمده‌است
جز
کنم
سپس
کنندگان
خودش
همواره
یافته
شان
صرف
نمی‌شود
رسیدن
چهارم
یابد
متر
ساز
داشته
کرده‌بود
باره
نحوه
کردم
تو
شخصی
داشته‌باشند
محسوب
پخش
کمی
متفاوت
سراسر
کاملا
داشتن
نظیر
آمده
گروهی
فردی
ع
همچون
خطر
خویش
کدام
دسته
سبب
عین
آوری
متاسفانه
بیرون
دار
ابتدا
شش
افرادی
می‌گویند
سالهای
درون
نیستند
یافته‌است
پر
خاطرنشان
گاه
جمعی
اغلب
دوباره
می‌یابد
لذا
زاده
,
:
؟
بي
.
تان
-
بی
يك
یا
براي
یک
را
'
یا
اين
كه
وا
ولي
دربراي
؟
که
گردد
اینجا""".split()
)

df1 = pd.read_excel ('i2.xlsx', dtype={'caption':str})
s = pd.Series(df1['caption'])

ws0 = s.str.cat(sep=' ')

fw = open("ws0.txt", "a")
fw.write(ws0+"\n")
fw.close()


ws0=ws0.replace('.',' در ')
ws0=ws0.replace('،',' در ')
ws0=ws0.replace(' «',' در ')
ws0=ws0.replace('» ',' در ')
ws0=ws0.replace(':',' در ')
ws0=ws0.replace('؛',' در')
ws0=ws0.replace('!',' در ')
ws0=ws0.replace('!',' در ')
ws0=ws0.replace('ايران','ايران-')
ws0=ws0.replace('ایران-','ايران')
ws0=ws0.replace('؟ ',' در')
#ws0=ws0.replace('»',' در ')
#ws0=ws0.replace(' ؟', ' در')

ws0 = [re.sub('[A-z]',' در ', sent) for sent in ws0]
ws0 = ''.join(ws0)

#f1 = open("other-text/before-stopped.txt", "a")

#f1.write(ws0)
#f1.close()
ws = ws0
a = []

#sss = set([',', 'و', 'که','ای','می','#','را','هم','در', 'از' , 'به', 'نگار', '،', 'این', '،', 'با', 'بر', 'تا', 'او', 'آن', 'یک', '؟', 'هر', 'یا', '!', ':', 'بار', 'اما', '»', '«', 'پس', 'بی', 'اگر' 'چه', 'که','مان', 'حتی', 'هیچ', 'چون' ])



for word in ws.lower().split():
    if word not in stopwords:
       a.append(word+' ')

#c = ''.join(a)

#aa = []
#for word in c.lower().split():
#    if word not in stopwords:
#       aa.append(word+' ')

b = ''.join(a)

fb = open("b.txt", "a")
fb.write(b+"\n")
fb.close()


normalizer = Normalizer()
datanrm = normalizer.normalize(b)
print('datanrm=', type(datanrm))

token = word_tokenize(datanrm)
print('token=', type(token))

tagger = POSTagger(model='resources/postagger.model')
tagdata = tagger.tag(word_tokenize(datanrm))
print('tagdata=', type(tagdata))

lemmatizer = Lemmatizer()
i = 0
dlm = []
for word in token:
   i = i + 1
   dlm.append([lemmatizer.lemmatize(word)])   


print('dlm=', type(dlm))

f1 = open("concatenated.txt", "a")
f1.write(ws+"\n")
f1.close()

f2 = open("normalized.txt", "a")
f2.write(datanrm+"\n")
f2.close()

f3 = open("token.txt", "a")
f3.write(str(token)+"\n")
f3.close()

f4 = open("tagdata.txt", "a")
f4.write(str(tagdata)+"\n")
f4.close()

with open('lem.txt', 'w') as filehandle:
    for listitem in dlm:
        filehandle.write('%s\n' % listitem)

# Create Dictionary
id2word = corpora.Dictionary(dlm)
print('id2word=', type(id2word))
print(id2word[0])
print(id2word[1])

# Create Corpus
# Term Document Frequency
corpus = [id2word.doc2bow(dl) for dl in dlm]
# View
print(corpus[:1])

with open('id2word.txt', 'w') as filehandle:
    for listitem in id2word:
        filehandle.write('%s\n' % listitem)

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5,
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           
                                           per_word_topics=True)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
f5 = open("val/topics.txt", "a")
f5.write(str(lda_model.print_topics())+"\n")
f5.close()

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=dlm, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Visualize the topics
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'val/LDA_Visualization.html')

