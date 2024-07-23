#include <algorithm>
#include <iostream>
using namespace std;

using int64 = long long;

int64 solve() {
  int64 A, B, C;
  cin >> A >> B >> C;

  // Watch out for this nasty bug: 2*x/y != 2*(x/y).
  // Remember to floor before multiplying!

  // Option 1: Buy only singles.
  int64 ans = C / A;

  // Option 2: Buy only doubles.
  ans = max(ans, 2*(C / B) - 1);

  // Option 3: 1 single + maximum doubles.
  if (C > A) {
    ans = max(ans, 1 + 2*((C - A)/B));
  }
  // Option 4: 2 singles + maximum doubles.
  if (C > 2*A) {
    ans = max(ans, 2 + 2*((C - 2*A)/B));
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
