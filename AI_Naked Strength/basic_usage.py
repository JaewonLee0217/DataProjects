## sqlite 엔진 생성 후 pandas dataframe으로 읽어오기
# DB 있을 떄, read_sql로 불러오기

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///investments.db')

df = pd.DataFrame({
    '날짜' : ['2021-01-01', '2021-01-02'],
    '투자액' : ['1000', '1500'],
    '수익' : ['100', '150']
})
df.to_sql('investments', engine, if_exists='replace', index=False)


data = pd.read_sql('SELECT * FROM investments', con=engine)
print(data)

