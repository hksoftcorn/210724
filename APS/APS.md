# APS

## 1. Dijkstra

```python
import heapq

N, K = 4, 2
times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
G = [[] for _ in range(N + 1)]
for u, v, w in times:
    G[u].append((v, w))

dist = [float('inf')] * (N + 1)
def dijkstra(start):
    Q = []
    heapq.heappush(Q, (0, start))
    dist[start] = 0
    
    while Q:
        d, u = heapq.heappop(Q)
        if dist[u] < d: continue
        
        for v, w in G[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
                heapq.heappush(Q, (dist[v], v))

dijkstra(K)
ans = max(dist[1:])
if ans == float('inf'):
    print(-1)
else:
    print(ans)
```

```python
import heapq

src, dst, k = 0, 2, 0
V = 3
edges = [[0, 1, 100], [1, 2, 100], [0, 2, 500]]
G = [[] for _ in range(V)]
for u, v, w in edges:
    G[u].append((v, w))

dist = [float('inf')] * V

def dijkstra(start):
    global k
    Q = []
    heapq.heappush(Q, (0, start))
    dist[start] = 0

    while Q and k >= 0:

        d, u = heapq.heappop(Q)
        if dist[u] < d: continue

        for v, w in G[u]:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w

        k -= 1        



dijkstra(src)
if dist[dst] == float('inf'):
    print(-1)
else:
    print(dist[dst])
```

## 2. Greedy

> 그리디 알고리즘은 글로벌 최적을 찾기 위해 각 단계에서 로컬 최적의 선택을 하는 휴리스틱 문제 해결 알고리즘입니다.

그리디 알고리즘이 잘 작동하는 문제들은 탐욕 선택 속성을 갖고 있는 최적 부분 구조인 문제들입니다. 여기서 탐욕 선택 속성이란 앞의 선택이 이후 선택에 영향을 주지 않는 것을 의미합니다. 다시 말해 그리디 알고리즘은 선택을 다시 고려하지 않습니다. 또한 최적 부분 구조란 문제의 최적 해결 방법이 부분 문제에 대한 최적 해결 방법으로 구성되는 경우를 말합니다. 

- 종류 : 다익스트라 알고리즘, 허프만 코딩 알고리즘, 의사결정 트리 알고리즘(ID3)
- 조건 : 1) 탐욕 선택 속성 2) 최적 부분 구조

### 2.1. 배낭 문제 Knapsack Problem

- 짐을 쪼갤 수 있는 경우(분할 가능 배낭 문제 => 그리디)
- 짐을 쪼갤 수 없는 경우(분할 불가능 배낭 문제 => DP)

```python
cargo = [[4, 12], [2, 1], [10, 4], [1, 1], [2, 2]]
def greedy_knapsack(cargo):
    capacity = 15
    pack = []
    
    for c in cargo:
        pack.append((c[1] / c[0], c[0], c[1]))
    pack.sort(reverse=True)
    
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
    return total_value
```

### 2.2. 주식

```python
stocks = [7, 1, 5, 3, 6, 4]
def greedy_stcok(stocks):
    result = 0
    for i in range(len(stocks) - 1):
        if stocks[i + 1] - stocks[i] > 0:
            result += stocks[i + 1] - stocks[i]
    return result

	# return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(stocks) - 1))
```



### 2.3. 키에 따른 대기열 구성

```python
import heapq
persons = [[7, 0], [4, 4], [7, 1], [5, 0], [6, 1], [5, 2]]
def heights(persons):
    Q = []
    for person in persons:
        heapq.heappush(Q, (-person[0], person[1]))
        
    result = []
    while Q:
        person = heapq.heappop(Q)
        result.insert(inser[1], [-perosn[0], person[1]])
   return result
```



### 2.4. 주유소

```python
gas = [1,2,3,4,5]
cost = [3,4,5,1,2]
```



### 2.5. 쿠키 나눔

```python
def cookieChild(children, cookies):
    children.sort()
    cookies.sort()
    i = j = 0
    while i < len(children) and j < len(cookies):
        if cookies[j] >= children[i]:
            i += 1
        j += 1
   return i
```



## 3. Divide & Conquer

### 3.1. 과반수 엘리트

```python
nums = [2,2,1,1,1,2,2]
def majorityElement(nums):
    counts = [0] * 1000000
    for num in nums:
        if counts[num] == 0:
            counts[num] = nums.count(num)
        if counts[num] > len(nums) // 2:
            return num

def majorityElement2(nums):
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    half = len(nums) // 2
    a = majorityElement2(nums[:half])
    b = majorityElement2(nums[half:])
    
    return [b, a][nums.count(a) > half]
```

### 3.2. 여러 방법의 연산

```python
def diffWaysCalculate(input_data):
    def compute(left, right, op):
        results = []
        for l in left:
            for r in right:
                result.append(eval(str(l) + op + str(r)))
        return results
    if input_data.isdisit():
        return [int(input_data)]
    
   	for idx, value in enumerate(input_data):
        if value in '-+*':
            left = diffWaysCalculate(input[:idx])
            right = diffWaysCalculate(input[idx + 1:])
            
            result.extend(compute(left, right, value))
    return results
        
```



## 4. DP

| 알고리즘         | 풀이 가능한 문제들의 특징                  | 풀이 가능한 문제 및 알고리즘                                 |
| ---------------- | ------------------------------------------ | ------------------------------------------------------------ |
| DP               | - 최적 부분 구조<br />- 중복된 하위 문제들 | - 0~1 배낭 문제<br />- 피보나치 수열<br />- 다익스트라 알고리즘 |
| Greedy           | - 최적 부분 구조<br />- 탐욕 선택 속성     | - 분할 가능 배낭 문제<br />- 다익스트라 알고리즘             |
| Divide & Conquer | - 최적 부분 구조                           | - 병합 정렬<br />- 퀵 정렬                                   |

- 최적 부분 구조

  서울에서 부산을 가려고 한다. 서울에서 대구가는 경로 중 최적의 경로의 길이 + 대구에서 부산에 가는 경로 중 최적의 경로의 길이를 더하면 서울에서 부산까지 가는 경로의 최소 거리가 나온다. 만약 서울에서 부산으로 직통도로가 개설된다면 더 이상 최적 부분 구조가 아니므로, DP와 Greedy 둘 다 적용이 불가능하다.

- DP 방법론은 크게 상향식과 하향식으로 나뉜다. 일반적으로 상향식을 타뷸레이션, 하향식을 메모이제이션이라고 부른다.
  - 타뷸레이션 : 작은 문제의 정답을 이용해 큰 문제의 정답을 풀어나간다.
  - 메모이제이션 : 하위 문제에 대한 정답을 계산했는지 확인해가며 문제를 풀어나간다.



### 4.1. 피보나치수열

```python
def fibo(n):
    if n <= 1:
        return n
    else:
        return fibo(n - 1) + fibo(n - 2)

# 타뷸레이션
def fibo(n):
    dp[0] = 0
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 메모이제이션
def fibo(n):
    if n <= 1:
        return n
    if dp[n]:
        return dp[n]
    dp[n] = fibo(n - 1) + fibo(n - 2)
    return dp[n]
```



### 4.2. 배낭문제

```python
cargo = [[4, 12], [2, 1], [10, 4], [1, 1], [2, 2]]
def greedy_knapsack(cargo):
    capacity = 15
    pack = []
    # 짐 갯수 : 5 + 1
    # 저장 가능한 용량 : 15 + 1
    # 6 x 16 배열 생성됨
    # 1열 전체, 1행 전체는 0으로 채워진다.
    for i in range(len(cargo) + 1):
        pack.append([])
        for c in range(capacity + 1):
            if i == 0 or c == 0:
                pack[i].append(0)
            elif cargo[i - 1][1] <= c:
                pack[i].append(
                    max(
                        cargo[i - 1][0] + pack[i - 1][c - cargo[i - 1][1]],
                        pack[i - 1][c]
                    ))
            else:
                pack[i].append(pack[i - 1][c])
                
	return pack[-1][-1]
    
    
```

