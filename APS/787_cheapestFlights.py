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