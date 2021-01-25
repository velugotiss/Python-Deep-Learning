word = list(input())
del word[3:5]
str1 = ''.join(word)[::-1]
print(str1)