path = './textdatasets/lotr_combined.txt'
text = open(path).read().lower()

allowed_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', ',', '-', '.', ' ', '!', '"', "'", '(', ')','/', '\n']
text = ''.join(filter(allowed_chars.__contains__, text))

tooManyLines = '\n\n\n\n\n'
tooManyLines2 = '\n\n\n\n'
tooManyLines3 = '\n\n\n'

path = './textdatasets/lotr_combined_cleaned.txt'
text = text.replace(tooManyLines, '\n')
text = text.replace(tooManyLines2, '\n')
text = text.replace(tooManyLines3, '\n')

with open('textdatasets/lotr_combined_cleaned.txt', 'w') as outfile:
  outfile.write(text)

outfile.close()