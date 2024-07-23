#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <vector>
using namespace std;

const int INF = (int)1e9, lgmax = 20;

int N, M, Q;
vector<vector<int>> adj, block_adj;

// Start of bridge-finding code.

int timer, nblocks;
vector<bool> visit;
vector<int> lowlink, tin, stack, block;
vector<vector<int>> blocks;
set<pair<int, int>> bridges;

void dfs(int u, int p) {
  visit[u] = true;
  lowlink[u] = tin[u] = timer++;
  stack.push_back(u);
  int children = 0;
  for (int v : adj[u]) {
    if (v == p) {
      continue;
    }
    if (visit[v]) {
      lowlink[u] = min(lowlink[u], tin[v]);
    } else {
      dfs(v, u);
      lowlink[u] = min(lowlink[u], lowlink[v]);
      if (lowlink[v] > tin[u]) {
        bridges.insert({u, v});
      }
      children++;
    }
  }
  if (lowlink[u] == tin[u]) {
    vector<int> tmp;
    int v;
    do {
      v = stack.back();
      stack.pop_back();
      tmp.push_back(v);
    } while (u != v);
    blocks.push_back(tmp);
  }
}

void tarjan(int nodes) {
  visit.assign(nodes, false);
  lowlink.assign(nodes, 0);
  tin.assign(nodes, 0);
  bridges.clear();
  stack.clear();
  blocks.clear();
  timer = 0;
  for (int i = 0; i < nodes; i++) {
    if (!visit[i]) {
      dfs(i, -1);
    }
  }
  // Get node to block mapping.
  block.assign(nodes, 0);
  nblocks = blocks.size();
  for (int i = 0; i < nblocks; i++) {
    for (int j : blocks[i]) {
      block[j] = i;
    }
  }
  // Get bridge-block forest adjacencies.
  block_adj.assign(nblocks, {});
  for (int i = 0; i < nodes; i++) {
    for (int j : adj[i]) {
      if (block[i] != block[j]) {
        block_adj[block[i]].push_back(block[j]);
      }
    }
  }
}

// Start of code to find odd cycles/distances to odd cycle blocks.

bool no_odd_cycles;
vector<int> color;
vector<bool> has_odd_cycle;
vector<int> dist_to_odd;

bool dfs_color(int u, int c = 1) {
  // Color edges in search of an odd cycle.
  color[u] = c;
  int nextc = 3 - c;
  for (int v : adj[u]) {
    if (bridges.count({u, v}) || bridges.count({v, u})) {
      continue;
    }
    if (color[v]) {
      if (color[v] != nextc) {
        return true;
      }
    } else if (dfs_color(v, nextc)) {
      return true;
    }
  }
  return false;
}

void find_odd_cycles_and_distances() {
  color.assign(N, 0);
  has_odd_cycle.assign(nblocks, false);
  dist_to_odd.assign(nblocks, INF);
  queue<int> q;
  for (int i = 0; i < nblocks; i++) {
    has_odd_cycle[i] = dfs_color(blocks[i].front());
    if (has_odd_cycle[i]) {
      dist_to_odd[i] = 0;
      q.push(i);
    }
  }
  no_odd_cycles = q.empty();
  while (!q.empty()) {
    int v = q.front();
    q.pop();
    for (int u : block_adj[v]) {
      if (dist_to_odd[u] == INF) {
        dist_to_odd[u] = dist_to_odd[v] + 1;
        q.push(u);
      }
    }
  }
}

// Start of binary lifting code for path-min queries.

vector<int> depth;
vector<vector<int>> lift, liftval;

void dfs_lift(int u, int p, int d) {
  depth[u] = d;
  lift[u][0] = p;
  liftval[u][0] = dist_to_odd[u];
  for (int v : block_adj[u]) {
    if (v != p) {
      dfs_lift(v, u, d + 1);
    }
  }
}

void init_binary_lifting() {
  depth.assign(nblocks, -1);
  lift.assign(nblocks, vector<int>(lgmax, -1));
  liftval.assign(nblocks, vector<int>(lgmax));
  dfs_lift(0, -1, 0);
  for (int e = 1; e < lgmax; e++) {
    for (int u = 0; u < nblocks; u++) {
      if (lift[u][e - 1] != -1) {
        lift[u][e] = lift[lift[u][e - 1]][e - 1];
        liftval[u][e] = min(liftval[u][e - 1], liftval[lift[u][e - 1]][e - 1]);
      }
    }
  }
}

int go_up(int u, int to_depth) {
  if (depth[u] == to_depth) {
    return u;
  }
  return go_up(lift[u][__builtin_ctz(depth[u] - to_depth)], to_depth);
}

int get_lca(int u, int v, int maxlift) {
  if (depth[u] != depth[v]) {
    if (depth[u] > depth[v]) {
      return get_lca(go_up(u, depth[v]), v, lgmax - 1);
    }
    return get_lca(u, go_up(v, depth[u]), lgmax - 1);
  }
  if (u == v) {
    return u;
  }
  if (lift[u][0] == lift[v][0]) {
    return lift[u][0];
  }
  while (lift[u][maxlift] == lift[v][maxlift]) {
    --maxlift;
  }
  return get_lca(lift[u][maxlift], lift[v][maxlift], maxlift);
}

int path_min_go_up(int u, int to_depth) {
  if (depth[u] == to_depth) {
    return dist_to_odd[u];
  }
  int e = __builtin_ctz(depth[u] - to_depth);
  return min(path_min_go_up(lift[u][e], to_depth), liftval[u][e]);
}

int path_min(int u, int v) {
  int lca = get_lca(u, v, lgmax - 1);
  return min(path_min_go_up(u, depth[lca]), path_min_go_up(v, depth[lca]));
}

// Main Solution.

long long solve() {
  cin >> N >> M;
  adj.assign(N, {});
  for (int i = 0, u, v; i < M; i++) {
    cin >> u >> v;
    --u, --v;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  // Find bridges and bridge block tree.
  tarjan(N);
  // Find odd cycle blocks, and min-distance to odd cycles from each block.
  find_odd_cycles_and_distances();
  // Initialize binary lifting for path mins.
  init_binary_lifting();
  // Process queries.
  long long ans = 0;
  cin >> Q;
  for (int i = 0, a, b; i < Q; i++) {
    cin >> a >> b;
    --a, --b;
    ans += no_odd_cycles ? -1 : path_min(block[a], block[b]);
  }
  return ans;
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
