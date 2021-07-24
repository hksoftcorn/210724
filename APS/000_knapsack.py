cargo = [[4, 12], [2, 1], [10, 4], [1, 1], [2, 2]]
def greedy_knapsack(cargo):
    capacity = 15
    pack = []
    
    for c in cargo:
        # $/kg, $, kg
        pack.append((c[0] / c[1], c[0], c[1]))
    pack.sort(reverse=True)
    print(pack)
    
    # 이렇게 float의 0을 정의할 수 있다니,,!! 정말 놀라운걸!!
    total_value: float = 0
    for p in pack:
        if capacity - p[2] >= 0:
            capacity -= p[2]
            total_value += p[1]
        else:
            fraction = capacity / p[2]
            total_value += p[1] * fraction
            break
        print(total_value)
    return total_value
print(greedy_knapsack(cargo))