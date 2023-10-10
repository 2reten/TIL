# Python Vis Project
## 사용한 모듈
```python
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
import re
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import nltk
import pickle
from nltk.corpus import stopwords
import re
nltk.download('all')
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
```
- chart_studio는 처음 사용해봤는데 링크를 걸어 ppt에서도 plotly의 기능을 사용할 수 있게 해줬다.
- nltk는 konltk와 같은 영어버전의 형태소 분석 모듈이다.

## 과정

### 웹 스크래핑
```python
driver = webdriver.Chrome()
```
- 먼저 driver를 webdriver모듈로 정의해 사용했다.
    - 원래 페이지를 자동 번역까지 하고 싶었으나 selenium의 우클릭이후에 마우스 조정을 실패했다. 하여 webdriver내 크롬에서 직접 설정으로 각 언어를 자동으로 영어 번역 시켰다.

```python
##### 한국 진보(조선일보)
kor_conservative_list = []
for j in range(1, 4):
    url = "https://www.chosun.com/nsearch/?query=%EC%98%A4%EC%97%BC%EC%88%98&page=" + str(j) + "&siteid=&sort=1&date_period=direct&date_start=20230820&date_end=20230927&writer=&field=&emd_word=&expt_word=&opt_chk=false&app_check=0&website=www,chosun&category="
    driver.get(url)
    time.sleep(1)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for i in range(1, 11):
        driver.find_elements("css selector", "#main > div.search-feed > div:nth-child("+str(i)+") > div > div.story-card.story-card--art-left.\|.flex.flex--wrap.box--hidden-md.box--hidden-lg > div.story-card-right.\|.grid__col--sm-9.grid__col--md-9.grid__col--lg-9.box--pad-left-xs > div.story-card__headline-container.\|.box--margin-bottom-xs > div > a > span")[0].click()
        time.sleep(1)
        ActionChains(driver).send_keys(Keys.END).perform()
        time.sleep(0.5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        if len(soup.select("section.article-body")[0].text) == 0:
            continue
        else:
            kor_conservative_list.append(soup.select("section.article-body")[0].text)
        driver.back()
        time.sleep(1)
```

-  페이지가 번역되는 과정에서 오류가 굉장히 많이 일어났다. 
왜 time.sleep를 이용해야 하는지 정확하게 배우는 계기가 되었다.
- ActionChains는 이번에 처음 사용한 모듈인데 ActionChains의 ()안에 변수를 넣으면 그 변수의 행동을 지정할 수 있다. 
이 코드를 이용해 페이지를 끝까지 내리거나 중간까지 내리는 동작을 수행했다.

```python
#####한국 프레시안(보수)
kor_progressive_list = []
for j in range(1, 4):
    url = "https://www.pressian.com/pages/search?sort=1&search=%EC%98%A4%EC%97%BC%EC%88%98&startdate=2023%EB%85%84%2008%EC%9B%94%2020%EC%9D%BC&enddate=2023%EB%85%84%2009%EC%9B%94%2027%EC%9D%BC&page=" + str(j)
    driver.get(url)
    time.sleep(1)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for i in range(10):
        driver.find_elements("css selector", "div.box")[i].click()
        time.sleep(1)
        ActionChains(driver).send_keys(Keys.END).perform()
        time.sleep(1)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for x in range(len(soup.select("div.article_body > p"))):
            kor_progressive_list.append(soup.select("div.article_body > p")[x].text)
        driver.back()
        time.sleep(1)
```

```python
##### 일본 아사히(보수)
jp_progressive_list = []
for j in range(0,41,20):
    url = "https://sitesearch.asahi.com/sitesearch/?Keywords=%E6%B1%9A%E6%9F%93%E6%B0%B4&Searchsubmit2=%E6%A4%9C%E7%B4%A2&Searchsubmit=%E6%A4%9C%E7%B4%A2&iref=pc_ss_date_btn1&sort=2&start=" + str(j)
    driver.get(url)
    time.sleep(1)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for i in range(1, 11):
        driver.find_elements("css selector", "#SiteSearchResult > li:nth-child("+str(i)+") > a > span.SearchResult_Headline > em > span")[0].click()
        time.sleep(1)
        ActionChains(driver).send_keys(Keys.END).perform()
        time.sleep(0.5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for x in range(len(soup.select("div.nfyQp > p"))):
            jp_progressive_list.append(soup.select("div.nfyQp > p")[x].text)
            
        time.sleep(3)
        driver.back()
        time.sleep(1)
```

```python
##### 일본 후쿠이(진보)
jp_conservative_list = []
driver.get("https://www.fukuishimbun.co.jp/search?fulltext=%E6%B1%9A%E6%9F%93%E6%B0%B4")
time.sleep(1)
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
for i in range(10):
    driver.find_elements("css selector", "div.article.clearfix > div.title")[i].click()
    time.sleep(1)
    ActionChains(driver).send_keys(Keys.END).perform()
    time.sleep(0.5)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for x in range(len(soup.select("div.article-body > p"))):
        jp_conservative_list.append(soup.select("div.article-body > p")[x].text)
    time.sleep(3)
    driver.back()
    time.sleep(1)
                   
for j in range(2,4):
    driver.get("https://www.fukuishimbun.co.jp/search?page=" + str(j) +"&fulltext=%E6%B1%9A%E6%9F%93%E6%B0%B4")
    time.sleep(1)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for x in range(10):
        driver.find_elements("css selector", "div.article.clearfix > div.title")[x].click()
        time.sleep(1)
        ActionChains(driver).send_keys(Keys.END).perform()
        time.sleep(0.5)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for x in range(len(soup.select("div.article-body > p"))):
            jp_conservative_list.append(soup.select("div.article-body > p")[x].text)
        time.sleep(3)
        driver.back()
        time.sleep(1)
```

```python
##### 러시아 Izvestiya(보수) - 검색어 (радиоактивная вода(방사능물))
rs_progressive_list = []
for j in range(3):
    driver.get("https://iz.ru/search?type=0&prd=0&from=" + str(j*10) +"&text=%D1%80%D0%B0%D0%B4%D0%B8%D0%BE%D0%B0%D0%BA%D1%82%D0%B8%D0%B2%D0%BD%D0%B0%D1%8F%20%D0%B2%D0%BE%D0%B4%D0%B0&date_from=2023-08-20&date_to=2023-09-27&sort=0")
    time.sleep(1)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    for i in range(3,13):
        driver.find_elements("css selector", "#block-purple-content > div:nth-child(" + str(i) + ") > div.view-search__title > a > font > font")[0].click()
        time.sleep(1)
        ActionChains(driver).send_keys(Keys.SPACE).perform()
        time.sleep(1)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        for x in range(len(soup.select("div.text-article > article > div.text-article__inside > div > div > p"))):
            rs_progressive_list.append(soup.select("div.text-article > article > div.text-article__inside > div > div > p")[x].text)
        time.sleep(3)
        driver.back()
        time.sleep(1)
```
- 러시아에서 문제가 발생했다. 페이지별로 기사가 두개가 존재했었는데 두 기사의 연관성이 전혀 없어 페이지를 끝까지 내리는것이 아닌 중간까지만 내려 페이지를 번역하고 다른 러시아어는 전처리 과정에서 날리는 방법을 이용했다.

```python
##### 중국 펑황왕신문(보수) - 검색어 (日本放射性水)
cn_progressive_list = []
driver.get("https://so.ifeng.com/?q=%E6%97%A5%E6%9C%AC%E6%94%BE%E5%B0%84%E6%80%A7%E6%B0%B4&c=1")
time.sleep(1)
driver.find_elements("css selector", "#root > div:nth-child(2) > div.index_content_rsXeq > div.index_tabBox_CwoGu > div > span:nth-child(2) > font > font")[0].click()
time.sleep(0.5)
ActionChains(driver).send_keys(Keys.END).perform()
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")
for i in range(30):
    driver.find_elements("css selector", "li.news-stream-newsStream-news-item-has-image.clearfix.news_item")[i].click()
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    try:
        driver.find_elements("css selector", "span.index_unfoldlcon_6tl7k")[i].click()
        ActionChains(driver).send_keys(Keys.END).perform()
    except:
        ActionChains(driver).send_keys(Keys.END).perform()
    time.sleep(0.5)
    for y in range(len(soup.select("div.index_detailBox_Jdkod > div > div > p"))):
        cn_progressive_list.append(soup.select("div.index_detailBox_Jdkod > div > div > p")[y].text)
    time.sleep(3)
```
- 중국의 언론사 사이트에서는 또 다른 오류가 발생했는데 기사를 읽으려면 "더 읽기"와 같은 버튼을 눌러야 했다.
하지만 모든 기사에 버튼이 있는것이 아니라 약간의 시행착오가 있었다.
    - try: except: 구문을 이용해 버튼이 있다면 클릭 없다면 오류가 발생하는 부분을 이용해 오류 발생시 페이지를 끝까지 내려 번역을 기다린 후 스크래핑을 하게 만들었다.

### 데이터 전처리

```python
with open("Kor_conservative_list", "w", encoding = "UTF-8") as f:
    for name in kor_conservative_list:
        f.write(name+",")
with open("Kor_progressive_list", "w", encoding = "UTF-8") as f:
    for name in kor_progressive_list:
        f.write(name+",")
with open("jp_conservative_list", "w", encoding = "UTF-8") as f:
    for name in jp_conservative_list:
        f.write(name+",")
with open("jp_progressive_list", "w", encoding = "UTF-8") as f:
    for name in jp_progressive_list:
        f.write(name+",")
with open("rs_progressive_list_1", "w", encoding = "UTF-8") as f:
    for name in rs_progressive_list:
        f.write(name+",")
with open("cn_progressive_list", "w", encoding = "UTF-8") as f:
    for name in cn_progressive_list:
        f.write(name+",")
```
- 파일을 텍스트 데이터로 저장하는 과정이다.

```python
f = open("Kor_conservative_list.txt", "r", encoding = "UTF-8")
kcs_data = f.read().split()
file.close()
f = open("Kor_progressive_list.txt", "r", encoding = "UTF-8")
kpr_data = f.read().split()
file.close()
f = open("jp_conservative_list.txt", "r", encoding = "UTF-8")
jpcs_data = f.read().split()
file.close()
f = open("jp_progressive_list.txt", "r", encoding = "UTF-8")
jppr_data = f.read().split()
file.close()
f = open("rs_progressive_list_1.txt", "r", encoding = "UTF-8")
rs_data = f.read().split()
file.close()
f = open("cn_progressive_list.txt", "r", encoding = "UTF-8")
cn_data = f.read().split()
file.close()
```

- 저장된 파일들을 여는 과정이다. 

```python
filtered_content = re.sub('[^,.?![A-Za-z\s]+','', str(kpr_data))
filtered_content = filtered_content.lower()
word_tokens = nltk.word_tokenize(filtered_content)
tokens_pos = nltk.pos_tag(word_tokens) 
NN_words2 = []

for word, pos in tokens_pos:
    if 'VB' in pos:
        NN_words2.append(word)
a = ["is", "am", "are", "was", "were", "be", "have", "has", "had", "been"]
kpr_nouns = [i for i in NN_words2 if i not in a]
df_kpr = pd.DataFrame({"word" : kpr_nouns})
df_kpr["count"] = df_kpr["word"].str.len()
df_kpr = df_kpr[df_kpr["count"] >= 2]
df_kpr.sort_values("count")
df_kpr = df_kpr.groupby("word", as_index = False).agg(n = ("word", "count")).sort_values("n", ascending = False)
kpr_top30 = df_kpr.head(30)
kpr_top30 = kpr_top30.reset_index(drop=True)
```
- 데이터의 전처리 과정이다. 특수문자들은 모두 날리고 대문자와 소문자, 공백만을 남겼다.
    - 그 후에 모든 단어를 소문자로 변환 후, nltk의 tokenize 함수를 이용해서 단어 옆에 그 단어의 형태소를 붙이는 방식을 이용했다.
    - for 문을 이용해서 단어와 형태소중 형태소의 값에 "VB"가 있는 값만을 리스트에 넣었고, 그 단어들로 데이터프레임을 만들고 단어가 한글자인 경우 만들 새로 저장했다.
    - count를 기준으로 정렬하고 word를 기준으로 그룹화한 후 agg함수를 이용해서 n을 그 단어가 사용된 횟수로 정의했다.
    - 그 후에 상위값 30개만을 출력했다.
```python
filtered_content = re.sub('[^,.?![A-Za-z\s]+','', str(kcs_data))
filtered_content = filtered_content.lower()
word_tokens = nltk.word_tokenize(filtered_content)
tokens_pos = nltk.pos_tag(word_tokens) 
NN_words = []

for word, pos in tokens_pos:
    if 'VB' in pos:
        NN_words.append(word)    
a = ["is", "am", "are", "was", "were", "be", "have", "has", "had", "been"]
kcs_nouns = [i for i in NN_words if i not in a]
df_kpr = pd.DataFrame({"word" : kpr_nouns})
df_kpr["count"] = df_kpr["word"].str.len()
df_kpr = df_kpr[df_kpr["count"] >= 2]
df_kpr.sort_values("count")
df_kpr = df_kpr.groupby("word", as_index = False).agg(n = ("word", "count")).sort_values("n", ascending = False)
kpr_top30 = df_kpr.head(30)
kpr_top30 = kpr_top30.reset_index(drop=True)
```
```python
filtered_content = re.sub('[^,?![A-Za-z\s]+','', str(jpcs_data))
filtered_content = filtered_content.lower()
word_tokens = nltk.word_tokenize(filtered_content)
tokens_pos = nltk.pos_tag(word_tokens) 
jp_words = []

for word, pos in tokens_pos:
    if 'VB' in pos:
        jp_words.append(word)
a = ["is", "am", "are", "was", "were", "be", "have", "has", "had", "been"]
jpcs_nouns = [i for i in jp_words if i not in a]
df_jpcs = pd.DataFrame({"word" : jpcs_nouns})
df_jpcs["count"] = df_jpcs["word"].str.len()
df_jpcs = df_jpcs[df_jpcs["count"] >= 2]
df_jpcs.sort_values("count")
df_jpcs = df_jpcs.groupby("word", as_index = False).agg(n = ("word", "count")).sort_values("n", ascending = False)
jpcs_top30 = df_jpcs.head(30)
```
```python
filtered_content = re.sub('[^,?![A-Za-z\s]+','', str(jppr_data))
filtered_content = filtered_content.lower()
word_tokens = nltk.word_tokenize(filtered_content)
tokens_pos = nltk.pos_tag(word_tokens) 
jp_words2 = []

for word, pos in tokens_pos:
    if 'VB' in pos:
        jp_words2.append(word)
a = ["is", "am", "are", "was", "were", "be", "have", "has", "had", "been"]
jppr_nouns = [i for i in jp_words2 if i not in a]
df_jppr = pd.DataFrame({"word" : jppr_nouns})
df_jppr["count"] = df_jppr["word"].str.len()
df_jppr = df_jppr[df_jppr["count"] >= 2]
df_jppr.sort_values("count")
df_jppr = df_jppr.groupby("word", as_index = False).agg(n = ("word", "count")).sort_values("n", ascending = False)
jppr_top30 = df_jppr.head(30)
jppr_top30 = jppr_top30.reset_index(drop=True)
```
```python
filtered_content = re.sub('[^,?![A-Za-z\s]+','', str(cn_data))
filtered_content = filtered_content.lower()
word_tokens = nltk.word_tokenize(filtered_content)
tokens_pos = nltk.pos_tag(word_tokens) 
cn_words = []

for word, pos in tokens_pos:
    if 'VB' in pos:
        cn_words.append(word)
a = ["is", "am", "are", "was", "were", "be", "have", "has", "had", "been"]
cn_nouns = [i for i in cn_words if i not in a]
df_cn = pd.DataFrame({"word" : cn_nouns})
df_cn["count"] = df_cn["word"].str.len()
df_cn = df_cn[df_cn["count"] >= 2]
df_cn.sort_values("count")
df_cn = df_cn.groupby("word", as_index = False).agg(n = ("word", "count")).sort_values("n", ascending = False)
cn_top30 = df_cn.head(30)
cn_top30 = cn_top30.reset_index(drop=True)
```
```python
filtered_content = re.sub('[^A-Za-z\s]+','', str(rs_data))
filtered_content = filtered_content.lower()
word_tokens = nltk.word_tokenize(filtered_content)
tokens_pos = nltk.pos_tag(word_tokens) 
rs_words = []

for word, pos in tokens_pos:
    if 'VB' in pos:
        rs_words.append(word)
a = ["is", "am", "are", "was", "were", "be", "have", "has", "had", "been"]
rs_nouns = [i for i in rs_words if i not in a]
df_rs = pd.DataFrame({"word" : rs_nouns})
df_rs["count"] = df_rs["word"].str.len()
df_rs = df_rs[df_rs["count"] >= 2]
df_rs.sort_values("count")
df_rs = df_rs.groupby("word", as_index = False).agg(n = ("word", "count")).sort_values("n", ascending = False)
rs_top30 = df_rs.head(30)
rs_top30 = rs_top30.reset_index(drop=True)
```

### 한국과 일본 데이터 합치기

```python
merged_df_kr = pd.concat([kcs_top30, kpr_top30], axis=1)
merged_df_jp = pd.concat([jpcs_top30, jppr_top30], axis=1)
```
- 시각화 하는 과정에서 편하게 사용하기 위해 진보와 보수의 데이터가 모두 있는 한국과 일본은 데이터를 합쳤다.

### 데이터 시각화

```python
username = ""
api_key = ""
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Bar(x=merged_df['kcs_word'], y=merged_df['kcs_count'], name='Korea conservative media company'), row = 1, col = 1)
fig.add_trace(go.Bar(x=merged_df['kcs_word'], y=merged_df['kpr_count'], name='Korea progressive media company'), row = 1, col = 1)
fig.add_trace(go.Bar(x=merged_df['kpr_word'], y=merged_df['kpr_count'], name='Korea progressive media company'), row = 2, col = 1)
fig.add_trace(go.Bar(x=merged_df['kpr_word'], y=merged_df['kcs_count'], name='Korea conservative media company'), row = 2, col = 1)
fig.update_layout(title_text='Comparison of Korean media companies', width=1000)
fig.show()

py.plot(fig, filename='Comparison of Korean media companies', auto_open=True)
```
- 한국의 진보 성향 언론사와 보수 성향 언론사를 비교했다.
    - 두개의 그래프를 만든 이유는 x축의 기준을 진보와 보수로 각각 만들어서 비교했기 때문이다.
```python
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Bar(x=merged_df['jpcs_word'], y=merged_df['jpcs_count'], name='Japan conservative media company'), row = 1, col = 1)
fig.add_trace(go.Bar(x=merged_df['jppr_word'], y=merged_df['jppr_count'], name='Japan progressive media company'), row = 1, col = 1)
fig.add_trace(go.Bar(x=merged_df['jppr_word'], y=merged_df['jppr_count'], name='Japan progressive media company'), row = 2, col = 1)
fig.add_trace(go.Bar(x=merged_df['jpcs_word'], y=merged_df['jpcs_count'], name='Japan conservative media company'), row = 2, col = 1)
fig.update_layout(title_text='Comparison of Japanese media companies', width=1000)
fig.show()
 
py.plot(fig, filename='Comparison of Japan media companies', auto_open=True)
```
```python
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Bar(x=kpr_top30['word'], y=kpr_top30['n'], name='Korea progressive media company'),row = 1, col = 1)
fig.add_trace(go.Bar(x=kpr_top30['word'], y=jppr_top30['n'], name='Japan progressive media company'),row = 1, col = 1)
fig.add_trace(go.Bar(x=jpcs_top30['word'], y=jpcs_top30['n'], name='Korea conservative media company'),row = 2, col = 1)
fig.add_trace(go.Bar(x=jpcs_top30['word'], y=kcs_top30['n'], name='Japan conservative media company'),row = 2, col = 1)
fig.update_layout(title_text='Comparison between Korea and Japan', width=1200)
fig.show()

py.plot(fig, filename='Comparison between Korea and Japan', auto_open=True)
```
- 두 나라의 진보 성향의 언론사와 보수 성향의 언론사끼리 비교했다.

```python
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Bar(x=kcs_top30['word'], y=kcs_top30['n'], name='Korea conservative media company'), row = 1, col = 1)
fig.add_trace(go.Bar(x=jpcs_top30['word'], y=jpcs_top30['n'], name='Japan conservative media company'), row = 2, col = 1)
fig.add_trace(go.Bar(x=cn_top30['word'], y=cn_top30['n'], name='China conservative media company'), row = 3, col = 1)
fig.add_trace(go.Bar(x=rs_top30['word'], y=rs_top30['n'], name='Russia conservative media company'), row = 4, col = 1)
fig.update_layout(title_text='Conservative media by country', width=1000)
fig.show()

py.plot(fig, filename='Conservative media by country_1', auto_open=True)
```
- 각나라의 단어 사용량을 시각화 했다.

```python
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Bar(x=kcs_top30['word'], y=kcs_top30['n'], name='Korea conservative media company'), row = 1, col = 1)
fig.add_trace(go.Bar(x=jpcs_top30['word'], y=jpcs_top30['n'], name='Japan conservative media company'), row = 2, col = 1)
fig.add_trace(go.Bar(x=cn_top30['word'], y=cn_top30['n'], name='China conservative media company'), row = 3, col = 1)
fig.add_trace(go.Bar(x=rs_top30['word'], y=rs_top30['n'], name='Russia conservative media company'), row = 4, col = 1)
fig.update_layout(title_text='Conservative media by country', width=1000)
fig.show()

py.plot(fig, filename='Conservative media by country_1', auto_open=True)
```

```python
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
fig = go.Figure()
fig.add_trace(go.Bar(x=kcs_top30['word'], y=kcs_top30['n'], name='Korea conservative media company'))
fig.add_trace(go.Bar(x=kcs_top30['word'], y=jpcs_top30['n'], name='Japan conservative media company'))
fig.add_trace(go.Bar(x=kcs_top30['word'], y=cn_top30['n'], name='China conservative media company'))
fig.add_trace(go.Bar(x=kcs_top30['word'], y=rs_top30['n'], name='Russia conservative media company'))
fig.update_layout(title_text='Conservative media by country', width=1000)
fig.show()

py.plot(fig, filename='Conservative media by country', auto_open=True)
```
- 각 나라의 보수 성향 데이터로 단어의 사용량을 비교했다.

### 감성 분석
- 원래는 머신러닝을 이용해서 만들고 싶었으나 일정의 조율을 실패해 머신러닝이 아닌 사전을 이용해서 코드를 작성했다.

```python
dic1 = open("positive-words.txt")
dic2 = open("negative-words.txt")
positive_words = []
negative_words = []
for line in dic1:
    positive_words.append(line.strip("\n"))
for line in dic2:
    negative_words.append(line.strip("\n"))
```

- 먼저 긍정적인 단어 리스트와 부정적인 단어의 리스트를 만들었다.

```python
kcs_sentiment_firm = []
sentiment = 0
count = 0
for token in kcs_nouns:
    if token in positive_words:
        sentiment += 1
        count += 1
    elif token in negative_words:
        sentiment -= 1
        count += 1
kcs_sentiment_firm.append(sentiment/count)
```
- 새로 감성 분석을 할 때 단어가 긍정적이라면 감성에 1점을 주고 count의 수를 1 증가, 부정적이라면 감성에서 1점을 빼고 count의 수를 1 증가시켰다.
    - 감성 점수를 count로 나눠서 점수를 리스트에 넣었다.

```python
kpr_sentiment_firm = []
sentiment = 0
count = 0
for token in kpr_nouns:
    if token in positive_words:
        sentiment += 1
        count += 1
    elif token in negative_words:
        sentiment -= 1
        count += 1
kpr_sentiment_firm.append(sentiment/count)
```

```python
jpcs_sentiment_firm = []
sentiment = 0
count = 0
for token in jpcs_nouns:
    if token in positive_words:
        sentiment += 1
        count += 1
    elif token in negative_words:
        sentiment -= 1
        count += 1
jpcs_sentiment_firm.append(sentiment/count)
```

```python
jppr_sentiment_firm = []
sentiment = 0
count = 0
for token in jppr_nouns:
    if token in positive_words:
        sentiment += 1
        count += 1
    elif token in negative_words:
        sentiment -= 1
        count += 1
jppr_sentiment_firm.append(sentiment/count)
```

```python
cn_sentiment_firm = []
sentiment = 0
count = 0
for token in cn_nouns:
    if token in positive_words:
        sentiment += 1
        count += 1
    elif token in negative_words:
        sentiment -= 1
        count += 1
cn_sentiment_firm.append(sentiment/count)
```

```python
rs_sentiment_firm = []
sentiment = 0
count = 0
for token in rs_nouns:
    if token in positive_words:
        sentiment += 1
        count += 1
    elif token in negative_words:
        sentiment -= 1
        count += 1
rs_sentiment_firm.append(sentiment/count)
```
### 감성 점수 데이터 시각화

```python
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
labels = ['kor_cons', 'kor_pr', 'p_cons', 'jp_pr', 'cn_cons', 'rs_cons']
values = [-0.405941, -0.730496, -0.582090, -0.366667, -0.365854, 0.076923]
# 그래프 생성
fig = px.bar(x=values, y=labels, orientation='h', text=values, color=values,
             labels={'x': 'Scores', 'y': 'Category'}, title="Sentiment Score")
# 레이아웃 설정
fig.update_layout(xaxis=dict(range=[-1, 1]), xaxis_title="Scores", yaxis_title="Category",
                  yaxis_categoryorder='total ascending', yaxis={'categoryarray': labels[::-1]})
# 그래프 출력
fig.show()
py.plot(fig, filename='Sentiment Score', auto_open=True)
```
- update_layout을 이용해서 x축의 범위를 -1 ~ 1로 설정하고 x축의 title을 Score로 설정했다.
- y축 역시 title을 Category로 설정하고 yaxis_categoryorder을 이용해서 y축 카테고리 순서를 설정했다. 'total ascending' 을 사용해서 y축 카테고리를 총계를 기준으로 오름차순 정렬했다.
- yaxis={'categoryarray': labels[::-1]}로 카테고리의 배열을 labels를 뒤집었다.

- 시각화 하는 과정에서 chart_studio를 사용한 이유는 ppt내에서 plotly의 장점을 사용하기 위해서 사용했다.
    - 만약 그 코드 내에서만 확인을 한다면 시각화 과정에서 chart_studio 부분의 코드는 반드시 하지 않아도 되는 부분이다.