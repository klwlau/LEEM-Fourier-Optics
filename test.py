k = range(10)

for i in k:
    temp = []
    for j in k:
        if i>=j:
            temp.append([i,j])
        else:
            break
    print(temp)