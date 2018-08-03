f = open('aha.txt', 'a+')
import sys
sys.stdout = f
print('aa')