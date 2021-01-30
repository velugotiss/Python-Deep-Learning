# open and read the file
input_file = open("C:\Spring 2021\python & DL\ICP2\input.txt")
wordcount = {}

# reading, splitting & Looping through the words in file
for word in input_file.read().split():
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1
print(wordcount)

# closing the file
input_file.close()

output = ''

# storing the output in file
for word, count in wordcount.items():
    output += word + ':' + str(count) + '\n'

output_file = open("C:\\Spring 2021\\python & DL\\ICP2\\output.txt", "w")
output_file.write(output)
output_file.close()