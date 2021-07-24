import heapq

N, K = 4, 2
times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
N, K = 8, 3
times = [[3, 1, 5], [3, 2, 2], [2, 1, 2], [3, 4, 1], [4, 5, 1], [5, 6, 1], [6, 7, 1], [7, 8, 1], [8, 1, 1]]

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