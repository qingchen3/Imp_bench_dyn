from _collections import defaultdict
import queue, math
import time, datetime
import sys
global n
global m
global max_level


def load_parameter(testcase):
    global n  # number of vertices
    global m  # number of edges
    global max_level

    if testcase == "SC":
        n = 66000000
    elif testcase == "trackers":
        n = 40421974
    elif testcase == "usa":
        n = 25000000
    elif testcase == "osmswitzerland":
        n = 30000000
    elif testcase == "youtube":
        n = 3200000
    elif testcase == "stackoverflow":
        n = 2600000
    elif testcase == "enron":
        n = 870000
    elif "power_law" in testcase or "path" in testcase or "star" in testcase or "random" in testcase or "complete" in testcase:
        m = 10000000
        if "complete" in testcase:
            n = int(math.sqrt(2 * m))
        elif "random" in testcase or "power" in testcase:
            n = m // 10
        else:
            n = 10000000
    max_level = math.ceil(math.log(n, 2))
    return



def loadGraph(testcase):
    global n
    global m
    global max_level

    edges = []
    if testcase == "test":
        n = 9
        m = 8
    if "power_law" in testcase or "path" in testcase or "star" in testcase or "random" in testcase or "complete" in testcase:
        lines = open("datasets/" + testcase, 'r').readlines()
        v_set = set()
        for line in lines:
            items = line.rstrip().split(' ')
            if items[0] == items[1]:
                continue
            t = int(items[2])
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, t])
            v_set.add(u)
            v_set.add(v)
        m = len(edges)
        n = len(v_set)
    elif testcase == "osmswitzerland":
        lines = open("raw_data/" + testcase, 'r').readlines()
        v_set = set()
        for line in lines:
            items = line.rstrip().split(',')
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            v_set.add(u)
            v_set.add(v)
            edges.append([u, v, int(items[2])])
        m = len(edges)
        n = len(v_set)
        #edges.sort(key=lambda x: x[2])
    elif testcase == "stackoverflow":
        lines = open("raw_data/" + testcase, 'r').readlines()
        for line in lines:
            items = line.rstrip().split()
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, int(items[2])])
        edges.sort(key = lambda x: x[2])
        n = 2600000
        m = len(edges)
    elif testcase == 'dnc':
        lines = open("datasets/" + testcase, 'r').readlines()
        for line in lines:
            items = line.rstrip().split(',')
            if items[0] == items[1]:
                continue
            t = int(items[2])
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, t])
        edges.sort(key = lambda x: x[2])
        m = len(edges)
        n = 1900
    elif testcase == 'enron':
        lines = open("datasets/" + testcase, 'r').readlines()
        for line in lines[1:]:
            items = line.rstrip().split()
            if items[0] == items[1]:
                continue
            t = int(items[3])
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, t])
        edges.sort(key = lambda x: x[2])
        m = len(edges)
        n = 870000
    elif testcase == 'trackers':
        lines = open("datasets/" + testcase, 'r').readlines()
        for line in lines[1:]:
            items = line.rstrip().split('\t')
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v])
        edges.sort(key = lambda x: x[2])
        m = len(edges)  # m = 140613762
        n = 40421974
    elif testcase == 'youtube':
        lines = open("raw_data/" + testcase, 'r').readlines()
        for line in lines:
            items = line.rstrip().split('\t')
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, int(time.mktime(datetime.datetime.strptime(items[2], "%Y-%m-%d").timetuple()))])
        edges.sort(key = lambda x: x[2])
        m = len(edges)
        n = 3200000
    elif testcase == 'tech':
        lines = open("datasets/" + testcase, 'r').readlines()
        for line in lines:
            if line.startswith('%'):
                continue
            items = line.rstrip().split()
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, int(items[3])])
        edges.sort(key = lambda x: x[2])
        m = len(edges)
        n = 34000
    elif testcase == 'wiki':
        lines = open("datasets/" + testcase, 'r').readlines()
        for line in lines:
            items = line.rstrip().split(" ")
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, int(items[3])])
        edges.sort(key = lambda x: x[2])
        m = len(edges)
        n = 7100
    elif testcase in ['fb', 'messages', 'call']:
        lines = open("datasets/" + testcase, 'r').readlines()
        for line in lines:
            items = line.rstrip().split(',')
            if items[0] == items[1]:
                continue
            u, v = order(int(items[0]), int(items[1]))
            edges.append([u, v, int(float(items[2]))])
        edges.sort(key = lambda x: x[2])
        m = len(edges)
        if testcase == 'fb':
            n = 899
        elif testcase == 'messages':
            n = 2000
        else:
            n = 7000

    max_level = math.ceil(math.log(n, 2))
    return edges, n, m


def constructST_adjacency_list(graph, n):
    #compute spanning tree
    st = defaultdict(set)
    visited = [False] * (n + 1)
    for u in range(1, n):
        if not visited[u]:
            visited[u] = True
            q = queue.Queue()
            if len(graph[u]) == 0:
                visited[u] = True
                continue
            q.put(u)
            while q.qsize() > 0:
                new_q = queue.Queue()
                while q.qsize() > 0:
                    x = q.get()
                    for y in graph[x]:
                        if not visited[y]:
                            st[x].add(y)
                            st[y].add(x)
                            new_q.put(y)
                            visited[y] = True
                q = new_q
    return st


# BFS on adjacency maxtrix
def BFS(graph, u, v):
    if u == v:
        return True

    N = len(graph)
    visited = [False] * len(graph)
    q = queue.Queue()
    q.put(u)
    visited[u] = True
    while q.qsize() > 0:
        new_q = queue.Queue()
        while q.qsize() > 0:
            x = q.get()
            if x == v:
                return True
            for i in range(1, N):
                if graph[x][i] != 0 and not visited[i]:
                    new_q.put(i)
                    visited[i] = True

        q = new_q

    return False


# BFS on adjacency list
def BFS_adj(graph, n, u, v):
    if u == v:
        return True

    visited = [False] * (n + 1)
    q = queue.Queue()
    q.put(u)
    visited[u] = True
    while q.qsize() > 0:
        new_q = queue.Queue()
        while q.qsize() > 0:
            x = q.get()
            if x == v:
                return True
            for i in graph[x]:
                if not visited[i]:
                    new_q.put(i)
                    visited[i] = True

        q = new_q

    return False


# compute the best BFS with minimum S_d
def best_BFS(graph, n, r):
    # BFS trees for one connected component

    # BFS to find all vertices
    vertices_set = set()  # vertices in one connected component
    visited = [False] * (n + 1)
    q = queue.Queue()
    q.put(r)
    visited[r] = True
    while q.qsize() > 0:
        x = q.get()
        vertices_set.add(x)
        for i in graph[x]:
            if not visited[i]:
                q.put(i)
                visited[i] = True

    minimum_dist = sys.maxsize
    target = None

    for u in vertices_set:
        distances = 0
        visited = [False] * (n + 1)
        q = queue.Queue()
        q.put(u)
        visited[u] = True
        level = 1
        while q.qsize() > 0:
            new_q = queue.Queue()
            while q.qsize() > 0:
                x = q.get()
                for i in graph[x]:
                    if not visited[i]:
                        new_q.put(i)
                        visited[i] = True
                        #st[x].add(i)
                        #st[i].add(x)
                        distances += level
            level += 1
            q = new_q
        if distances < minimum_dist:
            minimum_dist = distances
            #bfs_tree = st.copy()
            target = u
    return target


def order(a, b):
    if a < b:
        return a, b
    else:
        return b, a
