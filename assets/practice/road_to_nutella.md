You and your friends have drawn a really big connected graph in sidewalk chalk with \(N\) nodes (numbered from \(1..N\)) and \(M\) edges. \(Q\) times, there will be a race from node \(a_i\) to node \(b_i\)​, with a chance to win a coveted hazelnut chocolate snack. By the unbreakable rules of hop-scotch, everyone must travel along a path from node \(a_i\) to node \(b_i\) using edges in the graph, alternating which foot touches the ground at each node, starting each race with their left foot on \(a_i\).

Your friends will make a mad dash for the chocolate along the **shortest path** from \(a_i\) to \(b_i\)​. You on the other hand are looking for a more interesting challenge, and are allowed to take *any* path, potentially including any nodes (even \(b_i\)) multiple times. You want to end at node \(b_i\)​, but with the following conditions:
 - You must finish on a different foot from everyone who took the shortest path.
- To make things interesting, you'd like to minimize the number of edges you travel through more than once.

{{PHOTO_ID:903178538089777|WIDTH:600}}

*An illustration of the first sample. Your friends take a shortest path (blue), and you can take the path in red. The path in red travels through \(1\) edge multiple times: the edge connecting nodes \(6\) and \(8\).*

For each query, is it possible to fulfill your two conditions? If so, add the minimum number of edges you have to travel through multiple times to your answer. If not, add \(-1\) to your answer.

# Constraints

\(1 \leq T  \leq 140\)
\(1 \leq N, M, Q \leq 3*10^5\)

The sum of \(N\), \(M\), and \(Q\) over all cases are each no more than \(3*10^6\).
There will be at most one edge connecting the same pair of nodes directly.
The graph is connected.
No edge will connect a node with itself.
For all queries, \(a_i \neq b_i\). That is, the start and end of each race will be on different nodes.

# Input Format

Input begins with an integer \(T\), the number of test cases. For each test case, there is first a line containing two space-separated integers, \(N\) and \(M\). \(M\) lines follow, each containing two integers describing the endpoints of an edge. Then, there is a line containing a single integer \(Q\). \(Q\) lines follow, the \(i\)th of which contains two space-separated integers \(a_i\) and \(b_i\).

# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by the sum of answers to all queries in that case. The answer to a query is the minimum number of edges you'd need to travel through more than once to reach node \(b_i\) on a the other foot than your friends, or \(-1\) if it isn't possible to do so.

# Sample Explanation

The graph for the first case is shown above. The first query asks us to go from node \(7\) to node \(1\). The shortest paths (one shown in blue) will start on the left and end on the right foot, but we can end our our left by taking the red path. We'll cross the edge from \(6\) to \(8\) more than once, so the answer to this query is \(1\). The answers to the queries are \([1, 0, 2, 0, 2]\) respectively.

In the second case, the query answers are \([-1, -1]\) respectively.

In the third case, the query answers are \([0, 0]\) respectively.

In the fourth case, the query answers are \([-1, -1]\) respectively.



