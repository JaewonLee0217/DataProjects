'''
문제: 상하좌우 이동
N x N 크기의 정사각형 공간에서 주어진 이동 명령에 따라 캐릭터를 이동
'''

n = int(input())
plans = input().split()

x, y = 1, 1
dx = [0, -1, 0, 1]
dy = [1, 0, -1, 0]
move_types = ['L', 'R']