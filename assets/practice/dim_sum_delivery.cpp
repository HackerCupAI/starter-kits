#include <iostream>
using namespace std;

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": ";
    int R, C, A, B;
    cin >> R >> C >> A >> B;
    cout << (R > C ? "YES" : "NO") << endl;
  }
  return 0;
}
