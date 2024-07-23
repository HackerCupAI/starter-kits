#include <iostream>
using namespace std;

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": ";
    int S, D, K;
    cin >> S >> D >> K;
    int buns = 2*(S + D);
    int patties = S + 2*D;
    if (buns >= K + 1 && patties >= K) {
      cout << "YES" << endl;
    } else {
      cout << "NO" << endl;
    }
  }
  return 0;
}
