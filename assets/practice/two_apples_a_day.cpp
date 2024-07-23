#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

const int INF = (int)1e9 + 5;

int N, sz;
vector<int> a;

int try_sum(int sum) {
  int skipped = 0;
  int l = 0, r = sz - 1;
  int res = sum / 2;
  while (l <= r) {
    if (a[l] + a[r] == sum) {
      l++;
      r--;
      continue;
    }
    skipped++;
    if (a[l] + a[r] < sum) {
      res = sum - a[l++];
    } else {
      res = sum - a[r--];
    }
  }
  return (skipped <= 1 && res > 0) ? res : INF;
}

int solve() {
  cin >> N;
  sz = 2*N - 1;
  a.resize(sz);
  for (int i = 0; i < sz; i++) {
    cin >> a[i];
  }
  sort(a.begin(), a.end());
  int ans = min({
    try_sum(a[1] + a[sz - 1]), // remove first
    try_sum(a[0] + a[sz - 1]), // remove middle
    try_sum(a[0] + a[sz - 2])  // remove last
  });
  return ans == INF ? -1 : ans;
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
