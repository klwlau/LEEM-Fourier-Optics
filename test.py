k = range(10)

for i in k:
    temp = []
    temp2 =[]
    for j in k:
        if i>j:
            temp.append([i,j])
            temp2.append([j,i])
        else:
            break
    print(temp,temp2)