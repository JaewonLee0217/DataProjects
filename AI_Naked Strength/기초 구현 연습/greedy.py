'''
거스름돈으로 사용할 500원, 100원, 50원, 10원짜리 동전이 무한히 존재할 때,
거슬러 줘야 할 돈 N원을 동전 개수가 최소가 되도록 거슬러 주는 문제
'''

n = int(input())
count = 0
coin_types = [500, 100, 50, 10]

for coin in coin_types:
    count += n // coin
    n = n % coin

print(count)