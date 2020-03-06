def KnapsackS( n, v, w, available_w ):
    # stop condition
    # if we reach the last item to check, or if the the weight reduces to 0
    # condição de parada
    # se atingirmos o último item a verificar ou se o peso reduzir para 0
    if n < 0 or available_w <= 0:
        return 0
    # if I can't fit the item in the bag, skip it
    #se eu não conseguir colocar o item na bolsa, pule-o
    if w[n] > available_w:
        return KnapsackS( n-1, v, w, available_w )
    else:
        # solution
        # if the item fits in the bag
        # check if it is part of the solution or not            
        # solução
        # se o item couber na bolsa
        # verifique se faz parte da solução ou não    
        _max = max(
            # max value between
            # the current value NOT INCLUDED in solution vs
            # valor máximo entre
            # o valor atual NÃO INCLUÍDO na solução vs
            KnapsackS( n-1, v, w, available_w ),
            # the current value INCLUDED in solution
            # o valor atual INCLUÍDO na solução
            KnapsackS( n-1, v, w, available_w - w[n] ) + v[n]
        )
        return _max

n = 5
max_w = 10
v = [ 2, 5, 1, 4, 4 ]
w = [ 3, 2, 1, 2, 5 ]
r = KnapsackS( n-1, v, w, max_w )
print(r)