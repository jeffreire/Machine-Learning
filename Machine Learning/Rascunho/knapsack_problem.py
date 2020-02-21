# def backpack(total, itens):
#     matrix = [[0 for col in range (total + 1)] for row in range(len(itens[0]))]
#     for row in range(total + 1):
#         for col in range(total + 1):    
#             if itens[0][row] > col:
#                 matrix[row][col] = matrix[row - 1][col]
#             else:
#                 matrix[row][col] = max (matrix[row - 1][col], matrix[row - 1][col - itens[0][row] + itens[1][row]])
#             for i in range(len(itens[0])):
#                 print(matrix[i])
#     packed =[]
#     col = total
#     for row in range(len(itens[0])-1,-1,-1):
#         if row == 0 and matrix[row][col] != 0:
#             packed.insert(0, row)
#         if matrix[row][col] != matrix[row - 1][col]:
#             packed.insert(0, row)
#             col -= itens[0][row]
#     print(packed)
#     print('valor maximo:', matrix[len(itens[0])- 1][total])
# itensW = [12,1,2,1,4]
# itensV = [4,2,2,1,10]
# itens = [itensW, itensV]
# backpack(6,itens)

# l = []

# l.append([1, 3])
# l.append([0, 2])

# l.append([1])
# l.append([1, 3, 6])
# print(l)

from itertools import combinations

# def soma_lista(lista):
#     if len(lista) == 1:
#         return lista[0]
#     else:
#         return lista[0] +  soma_lista(lista[1:])


valuesList = [4,2,2,1,10]
pesoList = [12,1,2,1,4]
itens =[]
for i in range(len(valuesList)):
    itens.append({'peso': pesoList[i], 'value': valuesList[i]})
itens = sorted(itens, key= lambda k: k['peso'], reverse= True)
p = []
va = []
for i in itens:
    for v, c in i.items():
        if v == 'peso' or v == 'value':
            if v == 'peso' and c:
                p.append(c)
                if sum(p) > 15:
                    if v == 'peso': 
                        p.remove(c)         
            elif v == 'value' and c:
                if c:
                    va.append(c)
                if len(va) > len(p):
                    va.remove(c)
print('Peso por item: {} e Valor dos itens: {}'.format(p, sum(va)))