pattern = "ATCG"
f = file("hej.txt", 'r')
searchResult = []
i = 0
for row in f:
    s = row
    res = s.find(pattern)
    if res >= 0:
        searchResult.append(i)

    i = i + 1

print searchResult