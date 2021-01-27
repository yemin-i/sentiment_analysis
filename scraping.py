#!/usr/bin/env python
# coding: utf-8

# # 라이브러리 설정

# In[1]:


from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote
import pandas as pd


# # 2020년 9월 25일 기준 1위 영화 '테넷' 리뷰 스크래핑

# In[2]:


def get_movie_reviews(mcode, page_num = 10):
    
    movie_review_df = pd.DataFrame(columns = ['title', 'score', 'text'])
    url = 'https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=' + str(mcode) + '&target=after'
    idx = 0 # 인덱스 지정
    
    for _ in range(0, page_num):
        
        movie_page = urllib.request.urlopen(url).read()
        movie_page_soup = BeautifulSoup(movie_page, 'html.parser')
        review_list = movie_page_soup.find_all('td', {'class' : 'title'})
        
        for review in review_list:
        
            title = review.find('a', {'class' : 'movie color_b'}).get_text()  # 타이틀 가져오기
            score = review.find('em').get_text()  # 숫자로 된 평점 가져오기
            review_text = review.find('a', {'class' : 'report'}).get('href').split(',')[2]  # 텍스트 가져오기
            movie_review_df.loc[idx] = [title, score, review_text]
            idx += 1
            print('#', end = '')
        
        try:
            url = 'https://movie.naver.com' + movie_page_soup.find('a', {'class' : 'pg_next'}).get('href')
        except:
            break
            
    return movie_review_df


# In[4]:


movie_review_df = get_movie_reviews(190010, 1000)

movie_review_df.to_csv("tenet_review1.csv", header=True, index=False)


# # 2020년 9월 25일 기준 상영 영화 리뷰 스크래핑

# In[32]:


url = 'https://movie.naver.com/movie/point/af/list.nhn'
naver_movie = urllib.request.urlopen(url).read()
soup = BeautifulSoup(naver_movie, 'html.parser')
select = soup.find('select', {'id':'current_movie'})  # 현재 상영작 메뉴
movies = select.find_all('option')                    # 현재 상영작 리스트 가져오기

movies_dict = {}
for movie in movies[1:]:
    movies_dict[movie.get('value')] = movie.get_text() # 키는 mcode, 값은 영화명인 딕셔너리 구성
    
movie_review_df = pd.DataFrame(columns = ['title', 'score', 'text'])

for mcode in movies_dict:
    df = get_movie_reviews(mcode, 1) # 위에서 만든 함수로 review 가져오기
    movie_review_df = pd.concat([movie_review_df, df])
    
movie_review_df.to_csv("review.csv", header=True, index=False)


# In[ ]:




